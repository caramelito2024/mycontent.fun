import sys
import os
import asyncio
import qasync
import livepeer
from datetime import datetime
from dotenv import load_dotenv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QTextEdit, QPushButton, QLabel, QStackedWidget, 
                           QHBoxLayout, QFrame, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from swarmzero import Agent
from swarmzero.sdk_context import SDKContext
from swarmzero.swarm import Swarm
from livepeer.models import components
import logging

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LIVEPEER_API_KEY = os.getenv("LIVEPEER_API_KEY")

# Validate API keys
for key_name in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LIVEPEER_API_KEY"]:
    if not os.getenv(key_name):
        raise ValueError(f"Please set {key_name} in your .env file")

# Create SDK Context
sdk_context = SDKContext(config_path="./swarmzero_config.toml")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SocialMediaAPI:
    """Mock API for demonstration purposes"""
    async def get_trending_topics(self, platform):
        await asyncio.sleep(1)
        trends = {
            'twitter': [
                {'topic': '#AI', 'volume': '125K tweets', 'sentiment': 'positive'},
                {'topic': 'ChatGPT', 'volume': '89K tweets', 'sentiment': 'neutral'},
                {'topic': 'NFTs', 'volume': '67K tweets', 'sentiment': 'mixed'},
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
        self.trend_analyzer = Agent(
            name="Trend Analyzer",
            functions=[],
            instruction="""You are an expert social media trend analyzer. Your tasks are to:
            1. Analyze trending topics and their metrics
            2. Identify target demographics
            3. Understand sentiment and engagement patterns
            4. Suggest content opportunities
            Be specific and data-driven in your analysis.""",
            sdk_context=sdk_context,
            agent_id="trend_analyzer"
        )

    async def analyze_trend(self, trend_data):
        prompt = f"""Analyze this trending topic in detail:
        Topic: {trend_data['topic']}
        Metrics: {trend_data.get('volume', '') or trend_data.get('upvotes', '') or trend_data.get('views', '')}
        Sentiment: {trend_data.get('sentiment', 'unknown')}
        
        Provide:
        1. Trend Analysis: Why is this trending now?
        2. Demographics: Who is engaging with this trend?
        3. Engagement Patterns: How are people interacting with this topic?
        4. Content Opportunities: What type of content would perform well?
        5. Viral Potential: Rate the viral potential from 1-10 and explain why
        """
        response = await self.trend_analyzer.chat(prompt)
        return response

class MemeGenerator:
    def __init__(self, sdk_context):
        # Initialize SwarmZero agent
        self.meme_creator = Agent(
            name="Meme Creator",
            functions=[],
            instruction="""You are a creative meme generator specialized in viral content. Your tasks are to:
            1. Create engaging meme concepts based on trends
            2. Adapt content for different platforms
            3. Suggest viral hashtags
            4. Consider platform-specific features
            Be creative and understand internet culture and humor.""",
            sdk_context=sdk_context,
            agent_id="meme_generator"
        )
        
        # Initialize Livepeer client
        self.livepeer_client = livepeer.Livepeer(api_key=LIVEPEER_API_KEY)

    async def generate_meme_idea(self, trend_data, analysis):
        try:
            # Get meme concept from SwarmZero agent
            prompt = f"""Create an innovative meme concept for this trend:
            Topic: {trend_data['topic']}
            Analysis: {analysis}
            Platform Metrics: {trend_data.get('volume', '') or trend_data.get('upvotes', '') or trend_data.get('views', '')}

            Provide:
            1. Meme Format: Suggest the best meme template/format
            2. Text Content: Provide the exact text to use
            3. Visual Description: Describe the visual elements in detail
            4. Platform Adaptation: How to adapt for different platforms
            5. Hashtag Strategy: List of relevant hashtags
            6. Viral Elements: Explain why this will be shareable
            """
            concept_response = await self.meme_creator.chat(prompt)
            
            # Generate image using Livepeer AI
            image_prompt = self.extract_image_prompt(concept_response)
            image_result = await self.generate_meme_image(image_prompt, trend_data['topic'])
            
            return {
                'concept': concept_response,
                'image_url': image_result.get('url'),
                'metadata': image_result
            }
            
        except Exception as e:
            logger.error(f"Meme generation error: {str(e)}")
            return {'concept': concept_response, 'error': str(e)}

    def extract_image_prompt(self, concept_response):
        try:
            parts = concept_response.split("Visual Description:")
            if len(parts) > 1:
                visual_desc = parts[1].split("\n")[0].strip()
                return visual_desc
            return concept_response
        except Exception:
            return concept_response

    async def generate_meme_image(self, prompt, topic):
        try:
            req = components.NewAIImageGenerationPayload(
                prompt=f"Create a meme image: {prompt}",
                style="meme",
                model_id="stabilityai/stable-diffusion-2-1",
                num_inference_steps=50,
                guidance_scale=7.5
            )
            return await self.livepeer_client.ai.generate_image(req)
        except Exception as e:
            logger.error(f"Livepeer AI image generation failed: {str(e)}")
            raise

class ContentSwarm:
    def __init__(self, sdk_context):
        self.trend_analyzer = TrendAnalyzer(sdk_context)
        self.meme_generator = MemeGenerator(sdk_context)
        
        self.swarm = Swarm(
            name="Content Creation Team",
            description="A swarm of agents that collaborate on trend analysis and meme creation",
            instruction="Create viral-worthy content based on current trends",
            functions=[],
            sdk_context=sdk_context,
            agents=[
                self.trend_analyzer.trend_analyzer,
                self.meme_generator.meme_creator
            ]
        )

    async def process_trend(self, trend_data):
        analysis = await self.trend_analyzer.analyze_trend(trend_data)
        meme_result = await self.meme_generator.generate_meme_idea(trend_data, analysis)
        return analysis, meme_result

class TrendMemeWindow(QMainWindow):
    def __init__(self, loop=None):
        super().__init__()
        self.loop = loop
        self.api = SocialMediaAPI()
        self.content_swarm = ContentSwarm(sdk_context)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Trend & Meme Generator')
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Platform selection
        platform_layout = QHBoxLayout()
        platform_label = QLabel('Platform:')
        self.platform_combo = QComboBox()
        self.platform_combo.addItems(['twitter', 'reddit', 'tiktok'])
        fetch_button = QPushButton('Fetch Trends')
        fetch_button.clicked.connect(self.fetch_trends)
        
        platform_layout.addWidget(platform_label)
        platform_layout.addWidget(self.platform_combo)
        platform_layout.addWidget(fetch_button)
        platform_layout.addStretch()
        
        layout.addLayout(platform_layout)

        # Content areas
        content_layout = QHBoxLayout()
        
        # Trends Area
        trends_frame = self.create_content_frame("Trending Topics", "trends_area")
        content_layout.addWidget(trends_frame)
        
        # Analysis Area
        analysis_frame = self.create_content_frame("Trend Analysis", "analysis_area")
        content_layout.addWidget(analysis_frame)
        
        # Meme Area
        meme_frame = self.create_content_frame("Meme Concept & Image", "meme_area")
        content_layout.addWidget(meme_frame)
        
        layout.addLayout(content_layout)

    def create_content_frame(self, title, area_name):
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)
        
        label = QLabel(title)
        label.setFont(QFont('Arial', 12, QFont.Bold))
        layout.addWidget(label)
        
        text_area = QTextEdit()
        text_area.setReadOnly(True)
        text_area.setMinimumWidth(350)
        setattr(self, area_name, text_area)
        layout.addWidget(text_area)
        
        return frame

    @qasync.asyncSlot()
    async def fetch_trends(self):
        """Async method to fetch and analyze trends"""
        try:
            platform = self.platform_combo.currentText()
            self.trends_area.setText("Fetching trends...")
            self.analysis_area.setText("")
            self.meme_area.setText("")
            
            # Get trends
            trends = await self.api.get_trending_topics(platform)
            
            # Display trends
            trends_text = f"Trending on {platform.capitalize()}:\n\n"
            for trend in trends:
                metrics = trend.get('volume', '') or trend.get('upvotes', '') or trend.get('views', '')
                trends_text += f"📈 {trend['topic']} ({metrics})\n"
            self.trends_area.setText(trends_text)

            # Analyze first trend using the swarm
            if trends:
                self.analysis_area.setText("Analyzing trend...")
                self.meme_area.setText("Preparing meme concept...")
                
                analysis, meme_result = await self.content_swarm.process_trend(trends[0])
                
                self.analysis_area.setText(analysis)
                
                # Display both meme concept and generated image
                meme_text = f"Meme Concept:\n{meme_result['concept']}\n\n"
                if 'image_url' in meme_result:
                    meme_text += f"\nGenerated Image URL: {meme_result['image_url']}"
                if 'error' in meme_result:
                    meme_text += f"\nImage Generation Error: {meme_result['error']}"
                    
                self.meme_area.setText(meme_text)
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(error_msg)
            self.trends_area.setText(error_msg)

def main():
    try:
        app = QApplication(sys.argv)
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)

        window = TrendMemeWindow(loop)
        window.show()

        with loop:
            loop.run_forever()

    except Exception as e:
        print(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()