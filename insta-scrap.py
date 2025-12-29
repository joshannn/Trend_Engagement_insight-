import instaloader

target = input("Enter Instagram username: ").strip()

L = instaloader.Instaloader(
    download_pictures=True,
    download_videos=False,              
    download_video_thumbnails=False,
    save_metadata=False,               # metadata
    compress_json=False,
    post_metadata_txt_pattern=""       # .txt files
)

# Required for private accounts
# L.login("your_username", "your_password")

L.download_profile(target, profile_pic_only=False)

print("downloaded")
