# My Content.Fun

![Logo](./docs/logo.png)

## Overview

Welcome to **My Content.Fun**! 🎉 

#### -- This is a fun and interactive platform where you can explore trending topics, generate memes, and collaborate on grid trading strategies. Our vibrant UI, powered by **SwarmZero SDK** and APIs like **Livepeer** and **CoinGecko**, ensures you stay ahead of the trend while having fun along the way -- 

---

## Features

- **Trending Topics Tracker:** Fetch trends from Twitter, Reddit, and TikTok.
- **Meme Generator:** Create memes from trend data using Livepeer’s text-to-image API.
- **Token Tracker:** View top trending tokens and insights using CoinGecko.
- **Grid Trading Strategy:** Collaborate with Swarm agents to propose trading strategies.
- **GUI with PyQt5:** A sleek and colorful interface for an engaging experience.

---

## Prerequisites

1. **Python** 3.8 or higher

2. **API Keys:**
   - [Livepeer](https://livepeer.com/)
   - [MistralAI](https://mistralai.com/)
   - [CoinGecko](https://www.coingecko.com/)

3. **SwarmZero SDK Configuration.**

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd my-content-fun
   ``` 

2. Create a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
``` 

3. Install dependencies:

```pip install -r requirements.txt``` 

4. Add environment variables:

Create a .env file and include:

```
MISTRAL_API_KEY=your_mistral_api_key
LIVEPEER_API_KEY=your_livepeer_api_key
COINGECKO_API_KEY=your_coingecko_api_key
``` 

5. Configure SwarmZero SDK: Add swarmzero_config.toml in the root directory.

6. Start the Application

```rust 
python3 mycontent_fun.py
``` 


### Interface Features

- Platform Selector: Choose between Twitter, Reddit, or TikTok.
- Generate Meme: Click "Generate a DOPE Meme" to create memes from trending topics.
- Grid Strategy Execution: Use trending tokens to generate trading strategies.


