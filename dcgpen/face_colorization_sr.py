import cv2
import numpy as np
from skimage import io
from skvideo.io import FFmpegWriter
from tqdm.auto import tqdm
from pathlib import Path

import fsspec
from dcgpen import __init_paths
from dcgpen.align_faces import get_reference_facial_points, warp_and_crop_face
from dcgpen.face_detect.retinaface_detection import RetinaFaceDetection
from dcgpen.face_model.face_gan import FaceGAN
from dcgpen.face_parse.face_parsing import FaceParse
from dcgpen.sr_model.real_esrnet import RealESRNet
from kasane.utils.io import get_filesystem_from_path, generate_presigned_url_from_path


class FaceColorizationSR:
    def __init__(
        self,
        base_dir="./",
        in_size=1024,
        out_size=None,
        use_sr=True,
        model=None,
        channel_multiplier=2,
        narrow=1,
        key=None,
        device="cuda",
    ):
        self.facegan = FaceGAN(
            base_dir,
            in_size,
            out_size,
            model,
            channel_multiplier,
            narrow,
            key,
            device=device,
        )
        self.facedetector = RetinaFaceDetection(base_dir, device)
        self.threshold = 0.9
        self.in_size = in_size
        self.faceparser = FaceParse(base_dir, device=device)
        self.alpha = 1
        self.out_size = in_size if out_size is None else out_size
        self.use_sr = use_sr
        self.srmodel = RealESRNet(base_dir, "realesrnet", 2, 1024, device=device)
        self.kernel = np.array(
            ([0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]),
            dtype="float32",
        )

        # get the reference 5 landmarks position in the crop settings
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts = get_reference_facial_points(
            (self.in_size, self.in_size),
            inner_padding_factor,
            outer_padding,
            default_square,
        )

        # the mask for pasting restored faces back
        self.mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)

    def post_process(self, gray, out):
        out_rs = cv2.resize(out, gray.shape[:2][::-1])
        gray_yuv = cv2.cvtColor(gray, cv2.COLOR_BGR2YUV)
        out_yuv = cv2.cvtColor(out_rs, cv2.COLOR_BGR2YUV)

        out_yuv[:, :, 0] = gray_yuv[:, :, 0]
        final = cv2.cvtColor(out_yuv, cv2.COLOR_YUV2BGR)

        return final

    def mask_postprocess(self, mask, thres=26):
        mask[:thres, :] = 0
        mask[-thres:, :] = 0
        mask[:, :thres] = 0
        mask[:, -thres:] = 0
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        return mask.astype(np.float32)

    def process(self, img, aligned=False):
        orig_faces, enhanced_faces = [], []
        if aligned:
            ef = self.facegan.process(img)
            orig_faces.append(img)
            enhanced_faces.append(ef)

            if self.use_sr:
                ef = self.srmodel.process(ef)

            return ef, orig_faces, enhanced_faces

        facebs, landms = self.facedetector.detect(img)

        height, width = img.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            if faceb[4] < self.threshold:
                continue
            fh, fw = (faceb[3] - faceb[1]), (faceb[2] - faceb[0])

            facial5points = np.reshape(facial5points, (2, 5))

            of, tfm_inv = warp_and_crop_face(
                img,
                facial5points,
                reference_pts=self.reference_5pts,
                crop_size=(self.in_size, self.in_size),
            )

            if self.use_sr:
                img_sr = self.srmodel.process(of)
                if img_sr is not None:
                    of = cv2.resize(img_sr, of.shape[:2][::-1])

                # enhance the face
            ef = self.facegan.process(of)

            if of.shape[:2] != ef.shape[:2]:
                ef = self.post_process(of, ef)

            orig_faces.append(of)
            enhanced_faces.append(ef)

            # tmp_mask = self.mask
            tmp_mask = self.mask_postprocess(self.faceparser.process(ef)[0] / 255.0)
            tmp_mask = cv2.resize(tmp_mask, (self.in_size, self.in_size))
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)

            if min(fh, fw) < 100:  # gaussian filter for small faces
                ef = cv2.filter2D(ef, -1, self.kernel)

            ef = cv2.addWeighted(ef, self.alpha, of, 1.0 - self.alpha, 0.0)

            if self.in_size != self.out_size:
                ef = cv2.resize(ef, (self.in_size, self.in_size))
            tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)

            mask = tmp_mask - full_mask
            full_mask[np.where(mask > 0)] = tmp_mask[np.where(mask > 0)]
            full_img[np.where(mask > 0)] = tmp_img[np.where(mask > 0)]

        full_mask = full_mask[:, :, np.newaxis]
        # if self.use_sr and img_sr is not None:
        #     img = cv2.convertScaleAbs(img_sr*(1-full_mask) + full_img*full_mask)
        # else:
        img = cv2.convertScaleAbs(img * (1 - full_mask) + full_img * full_mask)

        return img, orig_faces, enhanced_faces


def read_write_video(class_object, filename: str, outputdict, cap, fps: str):
    """Read and write video

    Parameters
    ----------
    class_object: object of colorization class
    filename: video file path for writing
    outputdict: output dictionary parameters, i.e. how to encode the data when writing to file.
    cap: object class for video capturing from video file
    fps

    Returns
    -------
    none
    """
    writer = FFmpegWriter(
        filename,
        inputdict={"-r": fps},
        outputdict=outputdict,
    )
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as bar:
        while True:
            success, frame = cap.read()
            if not success:
                cap.release()
                writer.close()
                break
            img_out, _, _ = class_object.process(frame, aligned=False)
            writer.writeFrame(img_out[..., ::-1])
            bar.update()


def read_write_image_folger(class_object, input_path: str, save_path: str):
    """Read images from folder and save colorization images
    Parameters
    ----------
    class_object: object of colorization class
    input_path: input folder with images (local or s3)
    save_path: path to save colorization image
    Returns
    -------
    none
    """
    
    input_filesystem, _ = get_filesystem_from_path(input_path)
    fs = fsspec.filesystem(input_filesystem)
    image_list = sorted(fs.glob(f'{input_path}/*.*g'))
    
    for image in tqdm(image_list):
        if input_path.startswith("s3://"):
            image = generate_presigned_url_from_path(s3_client=None, path=image)
        frame_name = Path(image).stem
        frame = io.imread(image)[..., ::-1]
        img_out, _, _ = class_object.process(frame, aligned=False)
        cv2.imwrite(f'{save_path}/{frame_name}.png', img_out)