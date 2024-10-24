import sys
import os
import asyncio
import qasync
import livepeer
from dotenv import load_dotenv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QTextEdit, QPushButton, QLabel, QHBoxLayout,
                             QFrame, QComboBox, QInputDialog, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from swarmzero import Agent
from swarmzero.sdk_context import SDKContext
import logging
import requests

# Load environment variables
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
LIVEPEER_API_KEY = os.getenv("LIVEPEER_API_KEY")

if not MISTRAL_API_KEY or not LIVEPEER_API_KEY:
    raise ValueError("Missing API keys. Please set them in your .env file.")

# Initialize SDK context
sdk_context = SDKContext(config_path="./swarmzero_config.toml")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SocialMediaAPI:
    def __init__(self):
        self.custom_topics = []

    def add_custom_topic(self, topic_data):
        self.custom_topics.append(topic_data)

    async def get_trending_topics(self, platform):
        await asyncio.sleep(1)  # Simulate API delay
        trends = {
            'twitter': [
                {'topic': 'Cryptocurrency', 'volume': '125K tweets', 'sentiment': 'positive'},
                {'topic': 'Memecoins', 'volume': '89K tweets', 'sentiment': 'neutral'},
                {'topic': 'Trading', 'volume': '67K tweets', 'sentiment': 'mixed'},
            ]
        }
        return trends.get(platform, []) + self.custom_topics

class TrendAnalyzer:
    def __init__(self, sdk_context):
        self.agent = Agent(
            name="Trend Analyzer",
            functions=[],
            instruction="Return in 5 words a fun meme idea.",
            sdk_context=sdk_context,
            agent_id="trend_analyzer"
        )

    async def analyze_trend(self, trend_data):
        prompt = f"Analyze the topic '{trend_data['topic']}' with a volume of '{trend_data.get('volume')}' and sentiment '{trend_data['sentiment']}'."
        return await self.agent.chat(prompt)

class MemeGenerator:
    def __init__(self, sdk_context):
        self.agent = Agent(
            name="Meme Creator",
            functions=[],
            instruction="Create a fun meme.",
            sdk_context=sdk_context,
            agent_id="meme_generator"
        )
        self.livepeer_client = livepeer.Livepeer(api_key=LIVEPEER_API_KEY)

    async def generate_meme(self, trend_data, analysis):
        try:
            prompt = f"Generate a meme about {trend_data['topic']} with a {analysis} sentiment."
            response = requests.post(
                "https://dream-gateway.livepeer.cloud/text-to-image",
                headers={"Authorization": f"Bearer {LIVEPEER_API_KEY}"},
                json={"model_id": "ByteDance/SDXL-Lightning", "prompt": prompt, "height": 512, "width": 512}
            )
            response.raise_for_status()
            image_url = response.json().get("images", [{}])[0].get("url")
            if not image_url:
                raise ValueError("No image URL found.")
            return {"image_url": image_url}
        except Exception as e:
            logger.error(f"Meme generation error: {str(e)}")
            return {"error": str(e)}

class ContentSwarm:
    def __init__(self, sdk_context):
        self.analyzer = TrendAnalyzer(sdk_context)
        self.generator = MemeGenerator(sdk_context)

    async def process_trend(self, trend_data):
        analysis = await self.analyzer.analyze_trend(trend_data)
        meme = await self.generator.generate_meme(trend_data, analysis)
        return analysis, meme

class TrendMemeWindow(QMainWindow):
    def __init__(self, loop):
        super().__init__()
        self.api = SocialMediaAPI()
        self.swarm = ContentSwarm(sdk_context)
        self.loop = loop
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Trend & Meme Generator")
        self.setGeometry(100, 100, 1200, 800)

        layout = QVBoxLayout()
        layout.addLayout(self._create_platform_selector())
        layout.addLayout(self._create_content_area())

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def _create_platform_selector(self):
        layout = QHBoxLayout()
        self.platform_combo = QComboBox()
        self.platform_combo.addItems(['twitter'])

        fetch_button = QPushButton("Fetch Trends")
        fetch_button.clicked.connect(self.fetch_trends)

        custom_button = QPushButton("Add Custom Topic")
        custom_button.clicked.connect(self.add_custom_topic)

        layout.addWidget(QLabel("Platform:"))
        layout.addWidget(self.platform_combo)
        layout.addWidget(fetch_button)
        layout.addWidget(custom_button)
        layout.addStretch()
        return layout

    def _create_content_area(self):
        layout = QHBoxLayout()
        layout.addWidget(self._create_content_frame("Trending Topics", "trends_area"))
        layout.addWidget(self._create_content_frame("Trend Analysis", "analysis_area"))
        layout.addWidget(self._create_content_frame("Meme Concept & Image", "meme_area"))
        return layout

    def _create_content_frame(self, title, area_name):
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)
        layout.addWidget(QLabel(title, font=QFont("Arial", 12, QFont.Bold)))

        if area_name == "meme_area":
            content_area = QLabel()
            content_area.setTextInteractionFlags(Qt.TextBrowserInteraction)
            content_area.setOpenExternalLinks(True)
        else:
            content_area = QTextEdit()
            content_area.setReadOnly(True)

        setattr(self, area_name, content_area)
        layout.addWidget(content_area)
        return frame

    def add_custom_topic(self):
        try:
            topic, ok = QInputDialog.getText(self, "Add Custom Topic", "Enter topic name:")
            if not ok or not topic:
                return

            volume, ok = QInputDialog.getText(self, "Topic Volume", "Enter volume or upvotes:")
            if not ok:
                volume = "N/A"

            sentiment, ok = QInputDialog.getItem(self, "Sentiment", "Select sentiment:",
                                                 ["positive", "neutral", "mixed"], 0, False)
            if not ok:
                sentiment = "neutral"

            new_topic = {'topic': topic, 'volume': volume, 'sentiment': sentiment}
            self.api.add_custom_topic(new_topic)
            self.trends_area.append(f"Added: {topic} ({volume}) - {sentiment}")
        except Exception as e:
            logger.error(f"Error adding custom topic: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    @qasync.asyncSlot()
    async def fetch_trends(self):
        try:
            platform = self.platform_combo.currentText().lower()
            trends = await self.api.get_trending_topics(platform)
            self.trends_area.setText("\n".join([f"{t['topic']} ({t.get('volume', '')})" for t in trends]))

            if trends:
                analysis, meme = await self.swarm.process_trend(trends[0])
                self.analysis_area.setText(analysis)
                if "image_url" in meme:
                    image_url = meme["image_url"]
                    self.meme_area.setText(f"<a href='{image_url}'>{image_url}</a>")
                else:
                    self.meme_area.setText("Error generating meme.")
        except Exception as e:
            logger.error(f"Error fetching trends: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

def main():
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    window = TrendMemeWindow(loop)
    window.show()
    with loop:
        loop.run_forever()

if __name__ == "__main__":
    main()
