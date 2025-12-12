import requests

# Replace with your bot token and chat ID
TELEGRAM_BOT_TOKEN = "8292034134:AAHjiGjYGfivzW3IirfoyefhPFRKp3REAiw"
TELEGRAM_CHAT_ID = "1432479136"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("✅ Message sent successfully!")
    else:
        print(f"❌ Failed to send message: {response.text}")

if __name__ == "__main__":
    send_telegram_message("🚀 Test message from your YOLOV5 bot!")
