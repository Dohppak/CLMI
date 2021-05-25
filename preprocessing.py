import os
import glob
import torch
import random


def main():
    mp3_files = glob.glob('./dataset/**/*.mp3', recursive=True)
    random.shuffle(mp3_files)
    mp3_list = []
    for audio_path in mp3_files:
        thumbnail_path = audio_path.replace("audio","thumbnail").replace(".mp3",".jpg")
        if os.path.isfile(thumbnail_path):
            mp3_list.append(audio_path)

    tr_index = int(len(mp3_list)*0.8)
    va_index = int((len(mp3_list) - tr_index) *0.5)

    tr_list = mp3_list[:tr_index]
    va_list = mp3_list[tr_index: tr_index+va_index]
    te_list = mp3_list[tr_index+va_index:]
    if not os.path.exists(os.path.dirname("./dataset/split/tr_list.pt")):
        os.makedirs(os.path.dirname("./dataset/split/tr_list.pt"))

    torch.save(tr_list, "./dataset/split/tr_list.pt")
    print("train_list", len(tr_list))

    torch.save(va_list, "./dataset/split/va_list.pt")
    print("valid_list", len(va_list))

    torch.save(te_list, "./dataset/split/te_list.pt")
    print("test_list", len(te_list))


if __name__ == "__main__":
    main()