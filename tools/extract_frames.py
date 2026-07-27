import cv2
from pathlib import Path
from tqdm import tqdm


class FrameExtractor:

    def __init__(self, fps_extract=1):
        """
        fps_extract:
            Số frame muốn lấy mỗi giây.
            Ví dụ:
                1 = 1 ảnh/giây
                2 = 2 ảnh/giây
                5 = 5 ảnh/giây
        """
        self.fps_extract = fps_extract

    def extract(self, video_path, output_dir):

        video_path = Path(video_path)
        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise Exception(f"Không mở được video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # khoảng cách giữa 2 frame cần lấy
        interval = max(int(video_fps / self.fps_extract), 1)

        frame_index = 0
        image_index = 0

        pbar = tqdm(total=total_frames, desc=video_path.stem)

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_index % interval == 0:

                filename = f"{video_path.stem}_{image_index:05d}.jpg"

                save_path = output_dir / filename

                cv2.imwrite(str(save_path), frame)

                image_index += 1

            frame_index += 1

            pbar.update(1)

        cap.release()

        pbar.close()

        print(f"\nĐã trích {image_index} ảnh.")


def process_folder(
        input_folder,
        output_folder,
        fps_extract=1):

    extractor = FrameExtractor(fps_extract)

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    videos = []

    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        videos.extend(input_folder.glob(ext))

    print(f"Tìm thấy {len(videos)} video.\n")

    for video in videos:

        save_dir = output_folder / video.stem

        extractor.extract(video, save_dir)


if __name__ == "__main__":

    process_folder(
        input_folder="datasets/videos/1",
        output_folder="datasets/images/train/1",
        fps_extract=1
    )