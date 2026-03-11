import cv2
import imageio
import os

def gif_selector(video_path, output_gif, start_second=None, start_frame=None, end_frame=None, opt=False):
    tmp_frame_path = 'tmp_frame.jpg'
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    new_size = (width // 2, height // 2)
    recording = -1
    skip_second = 0
    gif = []

    if start_frame is None:
        start_frame = int(start_second * fps)

    for idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            end_frame = idx
            break

        if idx < start_frame:
            continue

        frame = cv2.resize(frame, new_size)
        cv2.imwrite(tmp_frame_path, frame)

        if not end_frame:
            if recording != -1:
                if idx % 3 == 0:
                    gif.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if skip_second > 0:
                skip_second -= 1
                continue

            decision = input()

            if decision == '':
                continue
            elif decision == 'r':
                gif.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                recording = idx
            elif decision == 's':
                skip_second = 60
                print(f"Skipping to frame {idx + skip_second}")
            elif decision == 'f':
                end_frame = idx
                break
        else:
            recording = start_frame
            if idx % 3 == 0:
                gif.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if idx >= end_frame:
                break
    
    print(f"Starting frame: {recording}")
    print(f"End frame: {end_frame}")
    
    if len(gif) > 0:
        imageio.mimsave(output_gif, gif, loop=0)
    
    if os.path.exists(tmp_frame_path):
        os.remove(tmp_frame_path)

    cap.release()

if __name__ == "__main__":
    import glob

    path = "results/task22"

    count = 0
    for track in glob.glob(os.path.join(path, '*')):
        # if count < 3 or count > 3:
        #     count+=1
        #     continue
        vid_path = os.path.join(track, "track.mp4")
        output_gif = os.path.join(track, "track.gif")

        gif_selector(vid_path, output_gif, start_frame=1000, end_frame=1180)
        count+=1