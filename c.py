import sys
import os
import asyncio
import qasync
import livepeer
from datetime import datetime
from dotenv import load_dotenv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QTextEdit, QPushButton, QLabel, QHBoxLayout,
                             QFrame, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from swarmzero import Agent
from swarmzero.sdk_context import SDKContext
from swarmzero.swarm import Swarm
from livepeer.models import components
import logging
from mistralai import Mistral
import requests


# Load environment variables and validate them
load_dotenv()

# API Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
LIVEPEER_API_KEY = os.getenv("LIVEPEER_API_KEY")

for key_name, key_value in [("MISTRAL_API_KEY", MISTRAL_API_KEY),
                            ("LIVEPEER_API_KEY", LIVEPEER_API_KEY)]:
    if not key_value:
        raise ValueError(f"{key_name} is missing. Please set it in your .env file.")

# Initialize SDK context
sdk_context = SDKContext(config_path="./swarmzero_config.toml")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = Mistral(api_key=MISTRAL_API_KEY)

class SocialMediaAPI:
    """Mock API to simulate fetching social media trends."""
    async def get_trending_topics(self, platform):
        await asyncio.sleep(1)
        trends = {
            'twitter': [
                {'topic': 'Cryptocurrency', 'volume': '125K tweets', 'sentiment': 'positive'},
                {'topic': 'Memecoins', 'volume': '89K tweets', 'sentiment': 'neutral'},
                {'topic': 'Trading', 'volume': '67K tweets', 'sentiment': 'mixed'},
            ],
            'reddit': [
                {'topic': 'Artificial Intelligence', 'upvotes': '45.2K', 'sentiment': 'positive'},
                {'topic': 'Machine Learning', 'upvotes': '32.1K', 'sentiment': 'positive'},
                {'topic': 'Programming', 'upvotes': '28.9K', 'sentiment': 'neutral'},
            ],
            'tiktok': [
                {'topic': '#AIart', 'views': '2.1M', 'sentiment': 'positive'},
                {'topic': '#coding', 'views': '1.8M', 'sentiment': 'positive'},
                {'topic': '#tech', 'views': '1.5M', 'sentiment': 'neutral'},
            ]
        }
        return trends.get(platform, [])

class TrendAnalyzer:
    def __init__(self, sdk_context):
        self.agent = Agent(
            name="Trend Analyzer",
            functions=[],
            instruction="""Based on the trends, return in 5 words a meme idea which is fun""",
            sdk_context=sdk_context,
            agent_id="trend_analyzer",
        )

    async def analyze_trend(self, trend_data):
        volume = trend_data.get('volume', 'unknown')
        prompt = f"Analyze the trending topic '{trend_data['topic']}' with a volume of '{volume}' and a sentiment of '{trend_data['sentiment']}'. Suggest opportunities for creating a viral meme based on this trend."
        return await self.agent.chat(prompt)

class MemeGenerator:
    def __init__(self, sdk_context):
        self.models = {
            "ByteDance/SDXL-Lightning": "ByteDance/SDXL-Lightning",
            "SG161222/RealVisXL_V4.0_Lightning": "SG161222/RealVisXL_V4.0_Lightning",
        }
        self.selected_model = list(self.models.values())[0]
        self.agent = Agent(
            name="Meme Creator",
            functions=[],
            instruction="""Create a fun meme""",
            sdk_context=sdk_context,
            agent_id="meme_generator"
        )
        self.livepeer_client = livepeer.Livepeer(api_key=LIVEPEER_API_KEY)

    def set_model(self, model_name):
        if model_name in self.models.values():
            self.selected_model = model_name

    async def generate_meme(self, trend_data, analysis):
        try:
            # Generate an image prompt based on the trend data and analysis
            prompt = f"Generate a meme about {trend_data['topic']} with a {analysis} sentiment."

            # Use the selected model for image generation
            model_id = self.selected_model

            # Make the API request to generate the image
            response = requests.post(
                "https://dream-gateway.livepeer.cloud/text-to-image",
                headers={"Authorization": f"Bearer {LIVEPEER_API_KEY}"},
                json={
                    "model_id": model_id,
                    "prompt": prompt,
                    "height": 512,
                    "width": 512
                }
            )
            response.raise_for_status()  # Raise exception for HTTP errors

            # Log the complete response for debugging
            logger.info(f"Livepeer Response: {response.json()}")

            # Extract the image URL
            image_url = response.json().get("images", [{}])[0].get("url")

            if not image_url:
                raise ValueError("No image URL found in the response.")

            logger.info(f"Generated Meme URL: {image_url}")
            return {"image_url": image_url}

        except Exception as e:
            logger.error(f"Meme generation error: {str(e)}")
            return {"error": str(e)}
            
    def _extract_image_prompt(self, concept):
        parts = concept.split("Visual Description:")
        return parts[1].strip() if len(parts) > 1 else concept

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
        self.generator = MemeGenerator(sdk_context)
        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self):
        self.setWindowTitle("Trend & Meme Generator")
        self.setGeometry(100, 100, 1200, 800)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addLayout(self._create_top_controls())
        layout.addLayout(self._create_content_area())

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def _apply_styles(self):
        # Main window style with gradient background
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #2E0854, stop:1 #8B4CA8);
            }
            
            QComboBox {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #4A148C, stop:1 #7B1FA2);
                color: white;
                border: 2px solid #9C27B0;
                border-radius: 5px;
                padding: 5px;
                min-width: 100px;
                font-weight: bold;
            }
            
            QComboBox:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #6A1B9A, stop:1 #9C27B0);
                border: 2px solid #CE93D8;
            }
            
            QComboBox::drop-down {
                border: none;
                background: #9C27B0;
                width: 30px;
            }
            
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #9C27B0, stop:1 #E040FB);
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #AB47BC, stop:1 #EA80FC);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            }
            
            QLabel {
                color: white;
                font-weight: bold;
            }
            
            QFrame {
                background: rgba(74, 20, 140, 0.5);
                border: 2px solid #9C27B0;
                border-radius: 10px;
            }
            
            QFrame:hover {
                border: 2px solid #CE93D8;
                background: rgba(74, 20, 140, 0.7);
            }
            
            QTextEdit {
                background: rgba(74, 20, 140, 0.3);
                color: white;
                border: 1px solid #9C27B0;
                border-radius: 5px;
                padding: 5px;
            }
            
            QTextEdit:hover {
                border: 1px solid #CE93D8;
            }
        """)

    def _create_top_controls(self):
        layout = QHBoxLayout()
        
        # Platform selection
        platform_layout = QHBoxLayout()
        platform_label = QLabel("Platform:")
        platform_label.setStyleSheet("font-size: 14px;")
        platform_layout.addWidget(platform_label)
        self.platform_combo = QComboBox()
        self.platform_combo.addItems(['twitter', 'reddit', 'tiktok'])
        platform_layout.addWidget(self.platform_combo)
        layout.addLayout(platform_layout)
        
        # Add some spacing
        layout.addSpacing(20)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        model_label.setStyleSheet("font-size: 14px;")
        model_layout.addWidget(model_label)
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(self.generator.models.keys()))
        self.model_combo.currentIndexChanged.connect(self.generator.set_model)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Add some spacing
        layout.addSpacing(20)
        
        # Fetch button
        fetch_button = QPushButton("Generate a DOPE Meme")
        fetch_button.clicked.connect(self.fetch_trends)
        fetch_button.setMinimumWidth(120)
        layout.addWidget(fetch_button)
        
        layout.addStretch()
        return layout

    def _create_content_area(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(20)

        # Create frames with consistent styling
        layout.addWidget(self._create_content_frame("Trending Topics", "trends_area"))
        layout.addWidget(self._create_content_frame("Trend Analysis", "analysis_area"))
        layout.addWidget(self._create_content_frame("Meme Concept & Image", "meme_area"))

        return layout

    def _create_content_frame(self, title, area_name):
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)
        
        # Create title label with enhanced styling
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            font-family: Arial;
            font-size: 16px;
            font-weight: bold;
            color: white;
            padding: 5px;
        """)
        layout.addWidget(title_label)

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

    @qasync.asyncSlot()
    async def fetch_trends(self):
        platform = self.platform_combo.currentText().lower()
        self.trends_area.setText("Fetching trends...")

        try:
            trends = await self.api.get_trending_topics(platform)

            if not trends:
                self.trends_area.setText(f"No trends found on {platform.capitalize()}.")
                return

            # Display trending topics
            self.trends_area.setText("\n".join([f"{t['topic']} ({t.get('volume', '')})" for t in trends]))

            # Process the first trend
            analysis, meme = await self.swarm.process_trend(trends[0])

            # Update analysis area
            self.analysis_area.setText(analysis)

            # Check if meme generation was successful
            if "image_url" in meme:
                image_url = meme["image_url"]
                self.meme_area.setText(f"Image: <a href='{image_url}'>{image_url}</a>")
                self.meme_area.setOpenExternalLinks(True)  # Enable clickable links
            else:
                error = meme.get("error", "Image not available.")
                self.meme_area.setText(f"Error: {error}")

        except Exception as e:
            self.trends_area.setText(f"Error fetching trends: {str(e)}")
            logger.error(f"Error fetching trends: {str(e)}")

def main():
    try:
        # Simple chat completion test
        response = client.chat.completions.create(
            model="mistral-small",  # Update the model name here
            messages=[{"role": "user", "content": "Say this is a test"}]
        )
        print("API Response:", response.choices[0].message.content)
    except Exception as e:
        print(f"Mistral API call failed: {e}")

    
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    window = TrendMemeWindow(loop)
    window.show()
    with loop:
        loop.run_forever()

if __name__ == "__main__":
    main()