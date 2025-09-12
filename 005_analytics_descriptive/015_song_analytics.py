import pandas as pd
import boto3
from googleapiclient.discovery import build

# 🎯 Objective: Fetch and analyze YouTube video data for ABBA remakes
# Data Source: YouTube Data API
# Setup YouTube Data API client and related key
# I store the key in AWS SSM Parameter Store for security - replace with your own method if needed

# Function to get API key from AWS SSM Parameter Store


def get_api_key_from_ssm(parameter_name, region_name='us-east-1'):
    ssm = boto3.client('ssm', region_name=region_name)
    response = ssm.get_parameter(Name=parameter_name, WithDecryption=True)
    return response['Parameter']['Value']

# Initialize YouTube Data API client
# Retrieve my Google API key securely from my SSM Account
API_KEY = get_api_key_from_ssm('/project/env/youtube_only_googleapis_key_001')  # Change parameter name as needed
#API = 'YOUR_API_KEY'
youtube = build('youtube', 'v3', developerKey=API_KEY)
print("YouTube Data API client created.")
# 🎯 List of video IDs for ABBA remakes
video_ids = [
    'Dq32mmYUelg',  # Sing it Live - Mamma Mia!
    'DFpW09Z7N98',  # ABBA - Mamma Mia (Live - ABBA Down Under)
    'ipi7ppPUSEs',  # Mamma Mia Medley
    'gbqQb_pmq50',  # Ripley Alexander Audition
    'Qn7ELG7tOhk',  # West End LIVE
    'b1SJMDnqhCo',  # Ripley Visualiser
    'unfzfe8f9NI',  # ABBA - Mamma Mia (Official Video)
]

def get_video_data(video_id):
    
    try:
        response = youtube.videos().list(
            part='snippet,statistics',
            id=video_id
        ).execute()
        if 'items' not in response or len(response['items']) == 0:
            print(f"No data found for video ID: {video_id}")
            return {
                'Title': 'Not Found',
                'Artist/Channel': 'Not Found',
                'Platform': 'YouTube',
                'Year': 'N/A',
                'Views': 0,
                'Performance Type': 'Unknown',
                'Region': 'Unknown',
                'Link': f"https://www.youtube.com/watch?v={video_id}"
            }
        item = response['items'][0]
        print("Fetched data for video ID:", video_id)
        return {
            'Title': item['snippet']['title'],
            'Artist/Channel': item['snippet']['channelTitle'],
            'Platform': 'YouTube',
            'Year': item['snippet']['publishedAt'][:4],
            'Views': int(item['statistics']['viewCount']),
            'Performance Type': 'Live/Studio',  # You can refine this manually
            'Region': 'Unknown',  # Optional: Add manually or via channel metadata
            'Link': f"https://www.youtube.com/watch?v={video_id}"
        }
    except Exception as e:
        error_message = str(e)
        if 'quotaExceeded' in error_message or 'quota' in error_message:
            print(f"API quota exceeded. Error: {error_message}")
            return {
                'Title': 'Quota Exceeded',
                'Artist/Channel': 'Quota Exceeded',
                'Platform': 'YouTube',
                'Year': 'N/A',
                'Views': 0,
                'Performance Type': 'Unknown',
                'Region': 'Unknown',
                'Link': f"https://www.youtube.com/watch?v={video_id}"
            }
        else:
            print(f"Error fetching data for video ID {video_id}: {error_message}")
            return {
                'Title': 'Error',
                'Artist/Channel': 'Error',
                'Platform': 'YouTube',
                'Year': 'N/A',
                'Views': 0,
                'Performance Type': 'Unknown',
                'Region': 'Unknown',
                'Link': f"https://www.youtube.com/watch?v={video_id}"
            }

# 📦 Collect data for all videos
video_data = [get_video_data(vid) for vid in video_ids]

# 📊 Convert to DataFrame and export to Excel
df = pd.DataFrame(video_data)
df.to_excel('output/abba_remakes.xlsx', index=False)

print("DataFrame created with video data:")
print("Columns in DataFrame:", df.columns.tolist())
print(df.describe())
print(df[['Title', 'Views']].head())
print("--")
