import json
import os
import youtube_dl
import wget
import argparse
import multiprocessing
from functools import partial
from contextlib import contextmanager

parser = argparse.ArgumentParser(description='thumbnail/meta crawling')
parser.add_argument('-a_dir', type=str, default='../dataset/audio')

def audio_crawl(url, audio_out_dir):
    ydl_opts = {
        'format': 'bestaudio/best',
        'writeinfojson': False,
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192'
        }],
        'postprocessor_args': [
            '-ar', '22050'
        ],
        'outtmpl': audio_out_dir
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download = True)

def _multi_crawl(url, audio_path):
    if "list" in url:
        pass
    else:
        try:
            audio_crawl(url, audio_path)
        except:
            pass

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def main():
    args = parser.parse_args()
    urls_dir = "../dataset/query"
    lang = ['en.json', 'ko.json', 'fr.json']
    save_path = []
    url_list = []
    for (root, dirs, files) in os.walk(urls_dir):
        for dir_name in dirs:
            print("start: ",dir_name)
            for la in lang:
                urls_dict = json.load(open(os.path.join(root, dir_name, la), 'r', encoding='utf-8')).keys()
                for url in urls_dict:
                    youtube_id = url.split("https://www.youtube.com/watch?v=")[-1]
                    audio_path = os.path.join(args.a_dir, dir_name, la.split(".json")[0], youtube_id)
                    if not os.path.exists(os.path.dirname(audio_path)):
                        os.makedirs(os.path.dirname(audio_path))
                    save_path.append(audio_path)
                    url_list.append(url)
    print(len(save_path), len(url_list))
    
    with poolcontext(processes = multiprocessing.cpu_count()-2) as pool:
        pool.starmap(_multi_crawl, zip(url_list, save_path))
    
if __name__ == '__main__':
    main()