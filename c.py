import sys
import os
import asyncio
import qasync
from datetime import datetime
from dotenv import load_dotenv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QTextEdit, QPushButton, QLabel, QStackedWidget, 
                           QHBoxLayout, QFrame, QComboBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
from swarmzero import Agent
from swarmzero.sdk_context import SDKContext

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

sdk_context = SDKContext(config_path="./swarmzero_config.toml")

class MockSocialMediaAPI:
    """Mock API for demonstration purposes"""
    def get_trending_topics(self, platform):
        trends = {
            'twitter': [
                {'topic': '#AI', 'volume': '125K tweets'},
                {'topic': 'ChatGPT', 'volume': '89K tweets'},
                {'topic': 'NFTs', 'volume': '67K tweets'},
            ],
            'reddit': [
                {'topic': 'Artificial Intelligence', 'upvotes': '45.2K'},
                {'topic': 'Machine Learning', 'upvotes': '32.1K'},
                {'topic': 'Programming', 'upvotes': '28.9K'},
            ],
            'tiktok': [
                {'topic': '#AIart', 'views': '2.1M'},
                {'topic': '#coding', 'views': '1.8M'},
                {'topic': '#tech', 'views': '1.5M'},
            ]
        }
        return trends.get(platform, [])

class ContentGenerator:
    def __init__(self):
        self.trend_analyzer = Agent(
            name="Trend Analyzer",
            instruction="You are an expert in analyzing social media trends and understanding viral content patterns.",
            sdk_context=sdk_context,
            functions=[]
        )
        
        self.meme_creator = Agent(
            name="Meme Creator",
            instruction="You are a creative meme generator that creates viral-worthy content based on trends.",
            sdk_context=sdk_context,
            functions=[]
        )

    async def analyze_trend(self, trend_data):
        prompt = f"""Analyze this trending topic and provide insights:
        Topic: {trend_data['topic']}
        Metrics: {trend_data.get('volume', '') or trend_data.get('upvotes', '') or trend_data.get('views', '')}
        
        Provide:
        1. Why this might be trending
        2. Key audience demographics
        3. Content opportunities
        """
        response = await self.trend_analyzer.chat(prompt)
        return response

    async def generate_meme_idea(self, trend_data, analysis):
        prompt = f"""Create a meme concept based on this trend:
        Topic: {trend_data['topic']}
        Analysis: {analysis}

        Provide:
        1. Meme format/template suggestion
        2. Text content
        3. Visual description
        4. Target audience
        5. Potential hashtags
        """
        response = await self.meme_creator.chat(prompt)
        return response

class TrendMemeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.content_generator = ContentGenerator()
        self.social_api = MockSocialMediaAPI()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Trend-Based Meme Generator')
        self.setMinimumSize(1000, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Create sidebar
        sidebar = self.create_sidebar()
        layout.addWidget(sidebar, stretch=1)

        # Create main content area
        main_content = self.create_main_content()
        layout.addWidget(main_content, stretch=4)

        self.set_dark_theme()

    def create_sidebar(self):
        sidebar = QFrame()
        sidebar.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(sidebar)

        platform_label = QLabel('Select Platform')
        platform_label.setStyleSheet('color: white; font-size: 14px;')
        self.platform_combo = QComboBox()
        self.platform_combo.addItems(['twitter', 'reddit', 'tiktok'])
        self.platform_combo.setStyleSheet('''
            QComboBox {
                background-color: #2C3E50;
                color: white;
                padding: 5px;
                border: none;
                border-radius: 5px;
            }
        ''')

        fetch_button = QPushButton('🔄 Fetch Trends')
        fetch_button.setStyleSheet('''
            QPushButton {
                background-color: #2980B9;
                color: white;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
            }
        ''')
        
        # Connect the button to the async handler using qasync
        fetch_button.clicked.connect(lambda: asyncio.create_task(self.fetch_trends()))

        layout.addWidget(platform_label)
        layout.addWidget(self.platform_combo)
        layout.addWidget(fetch_button)
        layout.addStretch()

        return sidebar

    def create_main_content(self):
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        # Trends area
        trends_label = QLabel('Trending Topics')
        trends_label.setStyleSheet('color: white; font-size: 18px; font-weight: bold;')
        self.trends_area = QTextEdit()
        self.trends_area.setReadOnly(True)
        self.trends_area.setStyleSheet('background-color: #2C3E50; color: white; border-radius: 5px;')

        # Analysis area
        analysis_label = QLabel('Trend Analysis')
        analysis_label.setStyleSheet('color: white; font-size: 18px; font-weight: bold;')
        self.analysis_area = QTextEdit()
        self.analysis_area.setReadOnly(True)
        self.analysis_area.setStyleSheet('background-color: #2C3E50; color: white; border-radius: 5px;')

        # Meme generation area
        meme_label = QLabel('Generated Meme Concept')
        meme_label.setStyleSheet('color: white; font-size: 18px; font-weight: bold;')
        self.meme_area = QTextEdit()
        self.meme_area.setReadOnly(True)
        self.meme_area.setStyleSheet('background-color: #2C3E50; color: white; border-radius: 5px;')

        layout.addWidget(trends_label)
        layout.addWidget(self.trends_area)
        layout.addWidget(analysis_label)
        layout.addWidget(self.analysis_area)
        layout.addWidget(meme_label)
        layout.addWidget(self.meme_area)

        return content_widget

    def set_dark_theme(self):
        self.setStyleSheet('''
            QMainWindow {
                background-color: #1A1A1A;
            }
            QFrame {
                background-color: #2C2C2C;
                border-radius: 10px;
            }
            QLabel {
                color: white;
            }
            QTextEdit {
                padding: 10px;
            }
        ''')

    async def fetch_trends(self):
        """Async method to fetch and analyze trends"""
        try:
            platform = self.platform_combo.currentText()
            self.trends_area.setText("Fetching trends...")
            self.analysis_area.setText("")
            self.meme_area.setText("")
            
            # Get trends
            trends = self.social_api.get_trending_topics(platform)
            
            # Display trends
            trends_text = f"Trending on {platform.capitalize()}:\n\n"
            for trend in trends:
                metrics = trend.get('volume', '') or trend.get('upvotes', '') or trend.get('views', '')
                trends_text += f"📈 {trend['topic']} ({metrics})\n"
            self.trends_area.setText(trends_text)

            # Analyze first trend
            if trends:
                self.analysis_area.setText("Analyzing trend...")
                analysis = await self.content_generator.analyze_trend(trends[0])
                self.analysis_area.setText(analysis)

                self.meme_area.setText("Generating meme concept...")
                meme_concept = await self.content_generator.generate_meme_idea(trends[0], analysis)
                self.meme_area.setText(meme_concept)
                
        except Exception as e:
            self.trends_area.setText(f"Error: {str(e)}")

async def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    try:
        import asyncio
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
        loop = asyncio.get_event_loop()
        
        # Create and show the main window
        window = TrendMemeWindow()
        window.show()
        
        # Run the event loop
        await qasync.QEventLoop(app).asyncio()
        
    except Exception as e:
        print(f"Error in main: {e}")
        
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error running main: {e}")