import sys
import httpx
import os
import asyncio
import requests
import qasync
import livepeer
from datetime import datetime
from dotenv import load_dotenv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QTextEdit, QPushButton, QLabel, QHBoxLayout,
                             QFrame, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from swarmzero import Agent
from swarmzero.sdk_context import SDKContext
from swarmzero.swarm import Swarm
from livepeer.models import components
import logging
from mistralai import Mistral


# Load environment variables and validate them
load_dotenv()

# API Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
LIVEPEER_API_KEY = os.getenv("LIVEPEER_API_KEY")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")

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

class ContentSwarm:
    def __init__(self, sdk_context):
        self.analyzer = TrendAnalyzer(sdk_context)
        self.generator = MemeGenerator(sdk_context)
        self.sdk_context = sdk_context

        # Initialize the agents
        self.data_collection_agent = Agent(
            name="Data Collection Agent",
            instruction="Collect price data on given tokens from various sources.",
            sdk_context=sdk_context,
            functions=[self.fetch_price_data_from_api]
        )
        
        self.analysis_agent = Agent(
            name="Data Analysis Agent",
            instruction="Analyze collected price data and generate insights.",
            sdk_context=sdk_context,
            functions=[self.analyze_price_data]
        )
        
        self.strategy_agent = Agent(
            name="Grid Strategy Agent",
            instruction="Propose a grid strategy based on the price analysis.",
            sdk_context=sdk_context,
            functions=[self.propose_grid_strategy_internal]
        )

        # Create the swarm
        self.swarm = Swarm(
            name="Trading Team",
            description="A swarm of agents collaborating on proposing grid strategies based on token prices",
            instruction="Collaborate on proposing grid strategies based on token prices",
            sdk_context=sdk_context,
            functions=[],
            agents=[self.data_collection_agent, self.analysis_agent, self.strategy_agent]
        )

    async def fetch_price_data_from_api(self, token_data):
        try:
            # Extract price data safely with fallbacks
            price = 0.0
            if isinstance(token_data, dict):
                if 'price' in token_data:
                    price = float(token_data['price'])
                elif 'data' in token_data and 'price' in token_data['data']:
                    price = float(token_data['data']['price'])
                
            return {
                "token": token_data.get("topic", "Unknown Token"),
                "price": price,
                "timestamp": datetime.now().isoformat(),
                "market_cap_rank": token_data.get("volume", "N/A"),
                "sentiment": token_data.get("sentiment", "neutral")
            }
        except Exception as e:
            logger.error(f"Error fetching price data: {str(e)}")
            return None
        
    async def process_trend(self, trend_data):
        """Process trend data and generate meme"""
        try:
            logger.info(f"Processing trend: {trend_data['topic']}")
            
            # Get trend analysis
            analysis = await self.analyzer.analyze_trend(trend_data)
            if isinstance(analysis, str):
                analysis_text = analysis
            else:
                analysis_text = await self.process_response(analysis)
            
            logger.info(f"Generated analysis: {analysis_text}")
            
            # Generate meme based on trend and analysis
            meme = await self.generator.generate_meme(trend_data, analysis_text)
            
            return analysis_text, meme
            
        except Exception as e:
            error_msg = f"Error processing trend: {str(e)}"
            logger.error(error_msg)
            return "Error in analysis", {"error": str(e)}

    async def analyze_price_data(self, price_data):
        if not price_data:
            return "No price data available for analysis"
        
        try:
            # Safely extract data with default values
            token = price_data.get("token", "Unknown Token")
            price = float(price_data.get("price", 0))
            market_cap_rank = price_data.get("market_cap_rank", "N/A")
            
            # Calculate metrics
            daily_range = {
                "high": price * 1.05,  # Estimated daily high (+5%)
                "low": price * 0.95,   # Estimated daily low (-5%)
                "volatility": "Medium"
            }
            
            # Generate detailed analysis
            analysis = (
                f"Analysis for {token}:\n"
                f"Current price: ${price:.6f}\n"
                f"Market Cap Rank: {market_cap_rank}\n"
                f"Estimated daily range: ${daily_range['low']:.6f} - ${daily_range['high']:.6f}\n"
                f"Volatility: {daily_range['volatility']}\n"
                f"Timestamp: {price_data.get('timestamp', 'N/A')}"
            )
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing price data: {str(e)}")
            return f"Error in price analysis: {str(e)}"

    async def propose_grid_strategy_internal(self, analysis):
        try:
            # Extract current price from analysis safely
            if "Current price: $" not in analysis:
                return "Unable to determine current price from analysis"
                
            try:
                price_str = analysis.split("Current price: $")[1].split("\n")[0]
                current_price = float(price_str)
            except (IndexError, ValueError) as e:
                logger.error(f"Error parsing price: {str(e)}")
                return "Error parsing price from analysis"
            
            # Calculate grid levels with safeguards
            grid_levels = 5
            price_range = 0.1  # 10% above and below current price
            
            if current_price <= 0:
                return "Invalid price: must be greater than 0"
            
            upper_bound = current_price * (1 + price_range)
            lower_bound = current_price * (1 - price_range)
            grid_spacing = (upper_bound - lower_bound) / max(1, grid_levels - 1)
            
            # Generate grid levels
            grid_prices = [lower_bound + i * grid_spacing for i in range(grid_levels)]
            
            # Format strategy output
            strategy = [
                "Grid Trading Strategy:",
                f"Number of grids: {grid_levels}",
                f"Upper bound: ${upper_bound:.6f}",
                f"Lower bound: ${lower_bound:.6f}",
                f"Grid spacing: ${grid_spacing:.6f}\n",
                "Grid Levels:"
            ]
            
            for i, price in enumerate(grid_prices, 1):
                strategy.append(f"Level {i}: ${price:.6f}")
                
            return "\n".join(strategy)
            
        except Exception as e:
            logger.error(f"Error proposing grid strategy: {str(e)}")
            return f"Error in strategy generation: {str(e)}"
    async def process_response(self, response):
        """Helper method to process responses whether they're streams, strings, or other types"""
        try:
            if response is None:
                return "No response received"
                
            if isinstance(response, str):
                return response
                
            if hasattr(response, 'read'):
                return await response.read()
                
            if hasattr(response, '__aiter__'):
                full_response = ""
                async for chunk in response:
                    if isinstance(chunk, str):
                        full_response += chunk
                    elif hasattr(chunk, 'content'):
                        full_response += chunk.content
                return full_response.strip()
                
            if hasattr(response, 'content'):
                return response.content
                
            return str(response)
            
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            return f"Error processing response: {str(e)}"

    async def propose_grid_strategy(self, token_data):
        try:
            # Log the incoming token data
            logger.info(f"Processing token data: {token_data}")
            
            # Collect price data
            price_data = await self.fetch_price_data_from_api(token_data)
            if not price_data:
                return "Failed to fetch price data", "Unable to generate strategy"

            logger.info(f"Fetched price data: {price_data}")

            # Get analysis
            analysis_response = await self.analysis_agent.chat(
                f"Analyze price data for {token_data.get('topic', 'Unknown Token')}"
            )
            analysis = await self.process_response(analysis_response)
            
            logger.info(f"Generated analysis: {analysis}")

            # Get strategy
            strategy_response = await self.strategy_agent.chat(
                f"Generate grid strategy based on analysis: {analysis}"
            )
            strategy = await self.process_response(strategy_response)
            
            logger.info(f"Generated strategy: {strategy}")

            return analysis, strategy

        except Exception as e:
            logger.error(f"Error in propose_grid_strategy: {str(e)}")
            return f"Error in analysis: {str(e)}", "Unable to generate strategy"

class SocialMediaAPI:
    """Mock API to simulate fetching social media trends."""
    async def get_trending_topics(self, platform):
        await asyncio.sleep(1)
        trends = {
            'twitter': [
                {'topic': 'Nature', 'volume': '125K tweets', 'sentiment': 'positive'},
                {'topic': 'Love', 'volume': '89K tweets', 'sentiment': 'positive'},
                {'topic': 'Space', 'volume': '67K tweets', 'sentiment': 'positive'},
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
    
    async def get_trending_search(self):
        url = "https://api.coingecko.com/api/v3/search/trending"
        headers = {"accept": "application/json"}
        params = {"limit": 5}  # Fetch the top 5 trending tokens
        try:
            response = requests.get(url, headers=headers, params=params, timeout=5)  # Add a timeout of 5 seconds
            data = response.json()
            print("Coingecko API response:", data)  # Add this line for logging
            if "coins" in data:
                return [{"topic": coin["item"]["name"], "volume": f"{coin['item']['market_cap_rank']}", "sentiment": "positive", "price": coin["item"]["data"]["price"], "thumb": coin["item"]["thumb"]} for coin in data["coins"]]
            else:
                return []
        except requests.exceptions.RequestException as e:
            print(f"Error fetching trending tokens: {str(e)}")
            return []

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

         # Meme style templates
        self.meme_styles = {
            "funny": "Create a humorous meme with witty text overlay, vibrant colors, and clear focal point.",
            "dramatic": "Design a dramatic meme with high contrast, intense emotions, and impactful visuals.",
            "minimalist": "Generate a clean, simple meme with strong visual hierarchy and minimal elements.",
            "retro": "Create a nostalgic meme with vintage aesthetics, classic meme format styling.",
            "modern": "Design a contemporary meme with sleek aesthetics and trending visual elements."
        }

    def _generate_meme_prompt(self, trend_data, analysis, style="funny"):
        """Generate an optimized prompt for meme creation"""
        base_style = self.meme_styles.get(style, self.meme_styles["funny"])
        
        sentiment = trend_data.get('sentiment', 'neutral').lower()
        topic = trend_data.get('topic', '')
        volume = trend_data.get('volume', '')

        # Enhanced prompt structure
        prompt_elements = [
            f"Create a high-quality meme image: {base_style}",
            f"Main subject: {topic}",
            "Style requirements:",
            "- Sharp, clear image quality",
            "- Modern meme aesthetic",
            "- Balanced composition",
            "- Proper lighting and contrast",
            f"Mood/Tone: {sentiment} and engaging",
            "Technical specifications:",
            "- High detail in focal points",
            "- Clean background",
            "- Professional quality render",
            f"Context: Trending topic with {volume} engagement",
            f"Analysis integration: {analysis[:100] if analysis else ''}"  # Limit analysis length
        ]

        return " | ".join(prompt_elements)

    async def generate_meme(self, trend_data, analysis):
        try:
            # Determine best style based on sentiment and topic
            style = "funny" if trend_data.get('sentiment') == 'positive' else "dramatic"
            
            # Generate optimized prompt
            prompt = self._generate_meme_prompt(trend_data, analysis, style)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://dream-gateway.livepeer.cloud/text-to-image",
                    headers={"Authorization": f"Bearer {LIVEPEER_API_KEY}"},
                    json={
                        "model_id": self.selected_model,
                        "prompt": prompt,
                        "height": 512,
                        "width": 512,
                        "num_inference_steps": 30,  # Increased for better quality
                        "guidance_scale": 7.5,      # Balanced guidance
                        "negative_prompt": "blurry, low quality, distorted, bad composition, poor lighting, unprofessional"
                    }
                )
                
            response.raise_for_status()
            data = response.json()
            image_url = data.get("images", [{}])[0].get("url")
            
            if not image_url:
                raise ValueError("No image URL found in the response.")
                
            return {
                "image_url": image_url,
                "prompt_used": prompt,  # For debugging
                "style_used": style     # For debugging
            }
            
        except Exception as e:
            logger.error(f"Meme generation error: {str(e)}")
            return {"error": str(e)}

    def set_model(self, model_name):
        if model_name in self.models.values():
            self.selected_model = model_name
            
    def _extract_image_prompt(self, concept):
        parts = concept.split("Visual Description:")
        return parts[1].strip() if len(parts) > 1 else concept

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

        # Add spacing
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

        # Add spacing
        layout.addSpacing(20)

        # Fetch button
        fetch_button = QPushButton("Generate a DOPE Meme")
        fetch_button.clicked.connect(self.fetch_trends)
        fetch_button.setMinimumWidth(120)
        layout.addWidget(fetch_button)

        # Add spacing
        layout.addSpacing(20)

        # Fetch trending tokens button
        fetch_tokens_button = QPushButton("Find Trending Tokens")
        fetch_tokens_button.clicked.connect(self.fetch_trending_tokens)
        fetch_tokens_button.setMinimumWidth(120)
        layout.addWidget(fetch_tokens_button)

        # Trading Advice button (disabled with hover text)
        trading_advice_container = QWidget()
        trading_advice_layout = QHBoxLayout(trading_advice_container)
        trading_advice_layout.setContentsMargins(0, 0, 0, 0)
        
        trading_advice_button = QPushButton("Trading Advice")
        trading_advice_button.setEnabled(False)  # Disable the button
        trading_advice_button.setToolTip("Coming Soon!")  # Add hover text
        trading_advice_button.setStyleSheet("""
            QPushButton:disabled {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                        stop:0 #666666, stop:1 #888888);
                color: #CCCCCC;
            }
        """)
        
        coming_soon_label = QLabel("(Coming Soon!)")
        coming_soon_label.setStyleSheet("""
            color: #FFD700;
            font-style: italic;
            padding-left: 5px;
        """)
        
        trading_advice_layout.addWidget(trading_advice_button)
        trading_advice_layout.addWidget(coming_soon_label)
        layout.addWidget(trading_advice_container)

        layout.addStretch()
        return layout

    def _create_content_area(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(20)

        # Create frames with consistent styling
        layout.addWidget(self._create_content_frame("Trending Topics", "trends_area"))
        layout.addWidget(self._create_content_frame("Meme Analysis", "analysis_area"))
        layout.addWidget(self._create_content_frame("Trending Tokens", "tokens_area"))
        layout.addWidget(self._create_content_frame("Trading Advice for Grid Strategy (SWARM)", "swarm_strategy_area"))
        layout.addWidget(self._create_content_frame("Meme Generation", "meme_area"))

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
            self.trends_area.setText("\n".join([f"{t['topic']} ({t.get('volume', '')})" for t in trends if 'topic' in t]))

            # Process the first trend
            analysis, meme = await self.swarm.process_trend(trends[0])

            # Update analysis area
            self.analysis_area.setText(analysis)

            # Check if meme generation was successful
            if "image_url" in meme:
                image_url = meme["image_url"]
                image_data = requests.get(image_url).content
                pixmap = QPixmap()
                pixmap.loadFromData(image_data)
                self.meme_area.setPixmap(pixmap)
            else:
                error = meme.get("error", "Image not available.")
                self.meme_area.setText(f"Error: {error}")

        except Exception as e:
            self.trends_area.setText(f"Error fetching trends: {str(e)}")
            logger.error(f"Error fetching trends: {str(e)}")

    @qasync.asyncSlot()
    async def fetch_trending_tokens(self):

        try:
            trending_tokens = await self.api.get_trending_search()

            if not trending_tokens:
                self.tokens_area.setText("No trending tokens found.")
                return

            # Display the top 5 trending tokens with unique buttons
            tokens_layout = QVBoxLayout()
            for token in trending_tokens[:5]:  # Display only the top 5 tokens
                price = token.get('price', 'N/A')  # Get the price if available, otherwise use 'N/A'
                if price != 'N/A':
                    price = round(price, 3)  # Round the price to 3 decimal places
                token_button = QPushButton(f"{token['topic']}: ${price}")
                token_button.clicked.connect(lambda checked=False, token=token: self.process_token(token))
                tokens_layout.addWidget(token_button)
            self.tokens_area.setLayout(tokens_layout)

        except Exception as e:
            self.tokens_area.setText(f"Error fetching trending tokens: {str(e)}")
            logger.error(f"Error fetching trending tokens: {str(e)}")

    @qasync.asyncSlot()
    async def run_swarm_execution(self):
        self.analysis_area.setText("Running swarm execution...")

        try:
            # Fetch the trending tokens
            trending_tokens = await self.api.get_trending_search()

            if not trending_tokens:
                self.analysis_area.setText("No trending tokens found.")
                return

            # Use the first token to run swarm analysis
            token_data = trending_tokens[0]
            self.tokens_area.setText(f"Using token: {token_data['topic']}")

            # Execute swarm with the token data
            analysis, strategy = await self.swarm.propose_grid_strategy(token_data)

            # Display swarm results in the analysis area
            self.analysis_area.setText(f"Swarm Analysis:\n{analysis}\n\nGrid Strategy:\n{strategy}")

        except Exception as e:
            logger.error(f"Error running swarm execution: {str(e)}")
            self.analysis_area.setText(f"Error: {str(e)}")


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