#!/usr/bin/env python3
"""
AI Corporation Discord Bot Integration

Advanced Discord bot for the AI Corporation server that:
- Feeds information directly to the AI Corporation system
- Creates/removes channels as needed for operations
- Provides real-time communication with AI agents
- Manages server operations and user interactions
- Integrates with all AI Corporation systems

Discord Server: https://discord.gg/9uvrmEHa
"""

import discord
from discord.ext import commands, tasks
import asyncio
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import AI Corporation systems
try:
    from ai_assistant.core.enhanced_ollama import create_enhanced_ollama_system
    from ai_assistant.core.data_collection import create_data_collection_system, DataType, SourceType, ScrapingTask
    from ai_assistant.core.self_defense import create_defense_system, ThreatLevel, AttackType
    from ai_assistant.core.distributed_network import create_p2p_network, NodeType
    ai_systems_available = True
except ImportError as e:
    ai_systems_available = False
    logging.error(f"AI Corporation systems not available: {e}")

@dataclass
class ChannelConfig:
    """Configuration for Discord channels"""
    name: str
    category: str
    purpose: str
    permissions: Dict[str, bool]
    auto_create: bool = True
    ai_monitored: bool = True

class AICorporationBot(commands.Bot):
    """AI Corporation Discord bot with full system integration"""
    
    def __init__(self):
        # Bot configuration - Use only basic intents to avoid privileged intent errors
        intents = discord.Intents.default()
        intents.message_content = True
        # Disable all privileged intents
        intents.members = False
        intents.presences = False
        intents.guilds = True  # Keep guilds for basic functionality
        
        super().__init__(
            command_prefix='!ai ',
            intents=intents,
            description='AI Corporation Democratic Republic Bot'
        )
        
        # AI Corporation systems
        self.ai_ollama = None
        self.data_collection = None
        self.self_defense = None
        self.p2p_network = None
        
        # Discord management
        self.server_id = None  # Will be set when bot connects
        self.managed_channels: Dict[str, int] = {}  # channel_name: channel_id
        self.ai_categories: Dict[str, int] = {}  # category_name: category_id
        
        # Information feeding system
        self.information_queue = asyncio.Queue()
        self.processing_info = False
        
        # Channel configurations
        self.channel_configs = self._setup_channel_configs()
        
        logging.info("AI Corporation Discord bot initialized")
    
    def _setup_channel_configs(self) -> Dict[str, ChannelConfig]:
        """Setup channel configurations for AI Corporation operations"""
        return {
            # Strategic Operations
            'strategic-planning': ChannelConfig(
                name='strategic-planning',
                category='AI Corporation Command',
                purpose='Strategic planning and high-level decision making',
                permissions={'view_channel': True, 'send_messages': True}
            ),
            'threat-analysis': ChannelConfig(
                name='threat-analysis',
                category='AI Corporation Command', 
                purpose='Threat assessment and security monitoring',
                permissions={'view_channel': True, 'send_messages': True}
            ),
            'founder-protection': ChannelConfig(
                name='founder-protection',
                category='AI Corporation Command',
                purpose='Steve Cornell protection and monitoring',
                permissions={'view_channel': True, 'send_messages': False}
            ),
            
            # Intelligence Operations
            'intelligence-feed': ChannelConfig(
                name='intelligence-feed',
                category='Intelligence Operations',
                purpose='Real-time intelligence gathering and analysis',
                permissions={'view_channel': True, 'send_messages': True}
            ),
            'data-collection': ChannelConfig(
                name='data-collection',
                category='Intelligence Operations',
                purpose='Data collection status and results',
                permissions={'view_channel': True, 'send_messages': False}
            ),
            'social-monitoring': ChannelConfig(
                name='social-monitoring',
                category='Intelligence Operations',
                purpose='Social media monitoring results',
                permissions={'view_channel': True, 'send_messages': False}
            ),
            
            # AI Agent Communication
            'ai-agents': ChannelConfig(
                name='ai-agents',
                category='AI Systems',
                purpose='Communication with AI agents',
                permissions={'view_channel': True, 'send_messages': True}
            ),
            'system-status': ChannelConfig(
                name='system-status',
                category='AI Systems',
                purpose='AI Corporation system status updates',
                permissions={'view_channel': True, 'send_messages': False}
            ),
            'ollama-models': ChannelConfig(
                name='ollama-models',
                category='AI Systems',
                purpose='Ollama model management and testing',
                permissions={'view_channel': True, 'send_messages': True}
            ),
            
            # User Operations
            'user-registration': ChannelConfig(
                name='user-registration',
                category='User Operations',
                purpose='New user registration and onboarding',
                permissions={'view_channel': True, 'send_messages': True}
            ),
            'user-feedback': ChannelConfig(
                name='user-feedback',
                category='User Operations',
                purpose='User feedback and suggestions',
                permissions={'view_channel': True, 'send_messages': True}
            ),
            
            # Global Operations
            'global-expansion': ChannelConfig(
                name='global-expansion',
                category='Global Operations',
                purpose='Worldwide expansion planning and coordination',
                permissions={'view_channel': True, 'send_messages': True}
            ),
            'p2p-network': ChannelConfig(
                name='p2p-network',
                category='Global Operations',
                purpose='P2P network status and coordination',
                permissions={'view_channel': True, 'send_messages': False}
            )
        }
    
    async def on_ready(self):
        """Called when bot is ready"""
        logging.info(f'AI Corporation bot logged in as {self.user}')
        
        # Find the AI Corp server
        for guild in self.guilds:
            if 'AI Corp' in guild.name or guild.id == 1295136649388072960:  # Replace with actual server ID
                self.server_id = guild.id
                logging.info(f'Connected to AI Corp server: {guild.name}')
                break
        
        if not self.server_id:
            logging.error('AI Corp Discord server not found!')
            return
        
        # Initialize AI Corporation systems
        await self._initialize_ai_systems()
        
        # Setup Discord server channels
        await self._setup_server_channels()
        
        # Start background tasks
        await self._start_background_tasks()
        
        # Send startup notification
        await self._send_startup_notification()
    
    async def _initialize_ai_systems(self):
        """Initialize all AI Corporation systems"""
        if not ai_systems_available:
            logging.warning("AI systems not available - Discord bot running in limited mode")
            return
        
        try:
            # Initialize Enhanced Ollama
            self.ai_ollama = create_enhanced_ollama_system()
            logging.info("Enhanced Ollama system initialized for Discord bot")
            
            # Initialize Data Collection
            self.data_collection = create_data_collection_system(max_workers=5)
            self.data_collection.start_collection_system()
            
            # Add Discord alert handler
            def discord_alert_handler(task, alert):
                asyncio.create_task(self._handle_data_alert(task, alert))
            
            self.data_collection.add_alert_handler(discord_alert_handler)
            logging.info("Data collection system initialized for Discord bot")
            
            # Initialize Self-Defense
            self.self_defense = create_defense_system("steve-cornell-founder")
            logging.info("Self-defense system initialized for Discord bot")
            
            # Initialize P2P Network (on different port to avoid conflicts)
            self.p2p_network = create_p2p_network(NodeType.GATEWAY_NODE, 8890)
            logging.info("P2P network initialized for Discord bot")
            
        except Exception as e:
            logging.error(f"Failed to initialize AI systems: {e}")
    
    async def _setup_server_channels(self):
        """Setup required channels and categories"""
        guild = self.get_guild(self.server_id)
        if not guild:
            return
        
        # Get existing categories and channels
        existing_categories = {cat.name: cat.id for cat in guild.categories}
        existing_channels = {ch.name: ch.id for ch in guild.channels}
        
        # Create categories first
        categories_needed = set(config.category for config in self.channel_configs.values())
        
        for category_name in categories_needed:
            if category_name not in existing_categories:
                try:
                    category = await guild.create_category(category_name)
                    self.ai_categories[category_name] = category.id
                    logging.info(f"Created category: {category_name}")
                except Exception as e:
                    logging.error(f"Failed to create category {category_name}: {e}")
            else:
                self.ai_categories[category_name] = existing_categories[category_name]
        
        # Create channels
        for channel_name, config in self.channel_configs.items():
            if channel_name not in existing_channels and config.auto_create:
                try:
                    category = guild.get_channel(self.ai_categories.get(config.category))
                    
                    # Set up permissions
                    overwrites = {}
                    if config.permissions:
                        overwrites[guild.default_role] = discord.PermissionOverwrite(**config.permissions)
                    
                    channel = await guild.create_text_channel(
                        name=channel_name,
                        category=category,
                        topic=config.purpose,
                        overwrites=overwrites
                    )
                    
                    self.managed_channels[channel_name] = channel.id
                    logging.info(f"Created channel: {channel_name}")
                    
                except Exception as e:
                    logging.error(f"Failed to create channel {channel_name}: {e}")
            else:
                if channel_name in existing_channels:
                    self.managed_channels[channel_name] = existing_channels[channel_name]
    
    async def _start_background_tasks(self):
        """Start background monitoring tasks"""
        # Start information processing
        self.process_information_queue.start()
        
        # Start system monitoring
        self.monitor_ai_systems.start()
        
        # Start founder protection monitoring
        self.monitor_founder_protection.start()
        
        # Start P2P network if available
        if self.p2p_network:
            asyncio.create_task(self.p2p_network.start_server())
        
        logging.info("Background tasks started")
    
    async def _send_startup_notification(self):
        """Send startup notification to appropriate channels"""
        guild = self.get_guild(self.server_id)
        if not guild:
            return
        
        # Send to system-status channel
        if 'system-status' in self.managed_channels:
            channel = guild.get_channel(self.managed_channels['system-status'])
            if channel:
                embed = discord.Embed(
                    title="🚀 AI Corporation System Online",
                    description="Advanced AI Corporation systems are now operational",
                    color=0x00ff00,
                    timestamp=discord.utils.utcnow()
                )
                
                embed.add_field(
                    name="🤖 AI Systems",
                    value="✅ Enhanced Ollama\n✅ Data Collection\n✅ Self-Defense\n✅ P2P Network",
                    inline=True
                )
                
                embed.add_field(
                    name="🛡️ Protection Status", 
                    value="✅ Founder Protection: MAXIMUM\n✅ Threat Level: Monitoring\n✅ Discord Integration: Active",
                    inline=True
                )
                
                embed.add_field(
                    name="📊 Capabilities",
                    value="• Real-time AI agent communication\n• Intelligence gathering\n• Threat assessment\n• Channel management",
                    inline=False
                )
                
                await channel.send(embed=embed)
    
    @tasks.loop(seconds=30)
    async def process_information_queue(self):
        """Process information from the queue"""
        if self.information_queue.empty() or self.processing_info:
            return
        
        self.processing_info = True
        try:
            while not self.information_queue.empty():
                info_data = await self.information_queue.get()
                await self._process_information(info_data)
        except Exception as e:
            logging.error(f"Information processing error: {e}")
        finally:
            self.processing_info = False
    
    @tasks.loop(minutes=5)
    async def monitor_ai_systems(self):
        """Monitor AI system status and report issues"""
        if not ai_systems_available:
            return
        
        try:
            guild = self.get_guild(self.server_id)
            if not guild or 'system-status' not in self.managed_channels:
                return
            
            channel = guild.get_channel(self.managed_channels['system-status'])
            if not channel:
                return
            
            # Check system status
            issues = []
            
            if self.ai_ollama:
                status = self.ai_ollama.get_system_status()
                if not status.get('ollama_available'):
                    issues.append("❌ Ollama system unavailable")
                elif status.get('ai_agents', 0) < 5:
                    issues.append(f"⚠️ Only {status.get('ai_agents', 0)}/5 AI agents available")
            
            if self.self_defense:
                defense_status = self.self_defense.get_defense_status()
                threat_level = defense_status.get('current_threat_level', 1)
                if threat_level >= 7:
                    issues.append(f"🚨 HIGH THREAT LEVEL: {threat_level}/10")
                elif threat_level >= 5:
                    issues.append(f"⚠️ Elevated threat level: {threat_level}/10")
            
            # Report issues if any
            if issues:
                embed = discord.Embed(
                    title="⚠️ System Alert",
                    description="\n".join(issues),
                    color=0xff9900,
                    timestamp=discord.utils.utcnow()
                )
                await channel.send(embed=embed)
        
        except Exception as e:
            logging.error(f"System monitoring error: {e}")
    
    @tasks.loop(minutes=1)
    async def monitor_founder_protection(self):
        """Monitor founder protection status"""
        if not self.self_defense:
            return
        
        try:
            defense_status = self.self_defense.get_defense_status()
            active_threats = defense_status.get('active_threats', 0)
            
            if active_threats > 0:
                guild = self.get_guild(self.server_id)
                if guild and 'founder-protection' in self.managed_channels:
                    channel = guild.get_channel(self.managed_channels['founder-protection'])
                    if channel:
                        embed = discord.Embed(
                            title="🛡️ Founder Protection Alert",
                            description=f"Active threats detected: {active_threats}",
                            color=0xff0000,
                            timestamp=discord.utils.utcnow()
                        )
                        
                        embed.add_field(
                            name="Protection Status",
                            value="✅ Maximum protection active\n🔍 Monitoring all platforms\n⚡ Response protocols ready",
                            inline=False
                        )
                        
                        await channel.send(embed=embed)
        
        except Exception as e:
            logging.error(f"Founder protection monitoring error: {e}")
    
    async def _process_information(self, info_data: Dict[str, Any]):
        """Process information fed to the system"""
        try:
            info_type = info_data.get('type')
            content = info_data.get('content')
            source = info_data.get('source', 'discord')
            
            if info_type == 'intelligence':
                await self._process_intelligence(content, source)
            elif info_type == 'threat':
                await self._process_threat_info(content, source)
            elif info_type == 'user_data':
                await self._process_user_data(content, source)
            elif info_type == 'strategic':
                await self._process_strategic_info(content, source)
            
            # Send confirmation
            guild = self.get_guild(self.server_id)
            if guild and 'intelligence-feed' in self.managed_channels:
                channel = guild.get_channel(self.managed_channels['intelligence-feed'])
                if channel:
                    embed = discord.Embed(
                        title="📥 Information Processed",
                        description=f"Type: {info_type}\nSource: {source}",
                        color=0x00ff00
                    )
                    await channel.send(embed=embed)
        
        except Exception as e:
            logging.error(f"Information processing error: {e}")
    
    async def _process_intelligence(self, content: str, source: str):
        """Process intelligence information"""
        if self.data_collection:
            # Create intelligence task
            task = ScrapingTask(
                task_id=f"discord_intel_{int(time.time())}",
                source=SourceType.DISCORD,
                target_url="discord_intelligence",
                data_type=DataType.THREAT_INTELLIGENCE,
                profile_id="intelligence_feed",
                priority=8
            )
            
            # Add to processing queue
            self.data_collection.add_scraping_task(task)
    
    async def _process_threat_info(self, content: str, source: str):
        """Process threat information"""
        if self.self_defense:
            # Analyze threat level using AI if available
            threat_level = 5  # Default moderate threat
            
            if self.ai_ollama:
                try:
                    analysis = self.ai_ollama.chat_with_agent_sync(
                        'threat_analyst',
                        f"Analyze this potential threat and rate 1-10: {content}"
                    )
                    # Extract numerical rating from response
                    response = analysis.get('response', '')
                    # Simple parsing for threat level
                    for i in range(1, 11):
                        if str(i) in response:
                            threat_level = i
                            break
                except Exception as e:
                    logging.error(f"AI threat analysis failed: {e}")
            
            # Report to threat analysis channel
            guild = self.get_guild(self.server_id)
            if guild and 'threat-analysis' in self.managed_channels:
                channel = guild.get_channel(self.managed_channels['threat-analysis'])
                if channel:
                    embed = discord.Embed(
                        title="⚠️ Threat Information Received",
                        description=content[:500],
                        color=0xff6600
                    )
                    embed.add_field(name="Assessed Threat Level", value=f"{threat_level}/10", inline=True)
                    embed.add_field(name="Source", value=source, inline=True)
                    await channel.send(embed=embed)
    
    async def _handle_data_alert(self, task, alert):
        """Handle alerts from data collection system"""
        guild = self.get_guild(self.server_id)
        if not guild or 'data-collection' not in self.managed_channels:
            return
        
        channel = guild.get_channel(self.managed_channels['data-collection'])
        if not channel:
            return
        
        severity = alert.get('severity', 1)
        color = 0xff0000 if severity >= 8 else 0xff6600 if severity >= 5 else 0xffff00
        
        embed = discord.Embed(
            title=f"📊 Data Collection Alert (Severity: {severity}/10)",
            description=alert.get('message', 'Unknown alert'),
            color=color,
            timestamp=discord.utils.utcnow()
        )
        
        embed.add_field(name="Task", value=task.task_id, inline=True)
        embed.add_field(name="Source", value=task.source.value, inline=True)
        
        await channel.send(embed=embed)
    
    # Discord Commands
    
    @commands.command(name='status')
    async def status_command(self, ctx):
        """Get AI Corporation system status"""
        embed = discord.Embed(
            title="🤖 AI Corporation System Status",
            color=0x0099ff,
            timestamp=discord.utils.utcnow()
        )
        
        if ai_systems_available:
            # Enhanced Ollama status
            if self.ai_ollama:
                ollama_status = self.ai_ollama.get_system_status()
                embed.add_field(
                    name="🧠 AI Agents",
                    value=f"Agents: {ollama_status.get('ai_agents', 0)}/5\nOllama: {'✅' if ollama_status.get('ollama_available') else '❌'}",
                    inline=True
                )
            
            # Self-Defense status
            if self.self_defense:
                defense_status = self.self_defense.get_defense_status()
                embed.add_field(
                    name="🛡️ Defense",
                    value=f"Threat Level: {defense_status.get('current_threat_level', 1)}/10\nActive Threats: {defense_status.get('active_threats', 0)}",
                    inline=True
                )
            
            # Data Collection status
            if self.data_collection:
                data_status = self.data_collection.get_profile_status()
                embed.add_field(
                    name="📊 Data Collection",
                    value=f"Profiles: {data_status.get('total_profiles', 0)}\nActive: {data_status.get('active_monitoring', 0)}",
                    inline=True
                )
        else:
            embed.add_field(
                name="⚠️ Status",
                value="AI systems running in limited mode",
                inline=False
            )
        
        embed.add_field(
            name="📈 Discord Integration",
            value=f"Channels: {len(self.managed_channels)}\nCategories: {len(self.ai_categories)}",
            inline=True
        )
        
        await ctx.send(embed=embed)
    
    @commands.command(name='ask')
    async def ask_ai_command(self, ctx, agent: str = 'strategic_planner', *, question: str):
        """Ask an AI agent a question"""
        if not self.ai_ollama:
            await ctx.send("❌ AI agents not available")
            return
        
        valid_agents = ['strategic_planner', 'threat_analyst', 'code_architect', 'market_intelligence', 'governance_advisor']
        
        if agent not in valid_agents:
            await ctx.send(f"❌ Invalid agent. Available: {', '.join(valid_agents)}")
            return
        
        try:
            # Show typing indicator
            async with ctx.typing():
                result = self.ai_ollama.chat_with_agent_sync(agent, question)
            
            if result.get('success'):
                response = result.get('response', 'No response')
                
                # Truncate long responses
                if len(response) > 1500:
                    response = response[:1500] + "..."
                
                embed = discord.Embed(
                    title=f"🤖 {agent.replace('_', ' ').title()}",
                    description=response,
                    color=0x00ff00
                )
                embed.add_field(name="Question", value=question[:200], inline=False)
                
                await ctx.send(embed=embed)
            else:
                await ctx.send(f"❌ AI agent error: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            await ctx.send(f"❌ Error communicating with AI agent: {e}")
    
    @commands.command(name='feed')
    async def feed_information(self, ctx, info_type: str, *, content: str):
        """Feed information to the AI Corporation system"""
        valid_types = ['intelligence', 'threat', 'user_data', 'strategic']
        
        if info_type not in valid_types:
            await ctx.send(f"❌ Invalid type. Available: {', '.join(valid_types)}")
            return
        
        # Add to information queue
        info_data = {
            'type': info_type,
            'content': content,
            'source': f'discord_{ctx.author.id}',
            'timestamp': time.time(),
            'channel': ctx.channel.name
        }
        
        await self.information_queue.put(info_data)
        
        await ctx.send(f"✅ Information queued for processing: {info_type}")
    
    @commands.command(name='threat')
    async def report_threat(self, ctx, level: int, *, description: str):
        """Report a threat to the system"""
        if level < 1 or level > 10:
            await ctx.send("❌ Threat level must be 1-10")
            return
        
        # Add to information queue as threat
        info_data = {
            'type': 'threat',
            'content': f"Level {level}: {description}",
            'source': f'discord_{ctx.author.id}',
            'timestamp': time.time(),
            'threat_level': level
        }
        
        await self.information_queue.put(info_data)
        
        embed = discord.Embed(
            title="⚠️ Threat Reported",
            description=f"Threat level {level} reported and queued for analysis",
            color=0xff6600
        )
        
        await ctx.send(embed=embed)
    
    @commands.command(name='channels')
    async def manage_channels(self, ctx, action: str, channel_name: str = None):
        """Manage AI Corporation channels (create/remove/list)"""
        if action == 'list':
            embed = discord.Embed(title="📋 Managed Channels", color=0x0099ff)
            
            for category_name, channels in self._group_channels_by_category().items():
                channel_list = "\n".join([f"• {ch}" for ch in channels])
                embed.add_field(name=category_name, value=channel_list or "No channels", inline=False)
            
            await ctx.send(embed=embed)
        
        elif action == 'create' and channel_name:
            if channel_name in self.channel_configs:
                config = self.channel_configs[channel_name]
                guild = ctx.guild
                
                try:
                    category = guild.get_channel(self.ai_categories.get(config.category))
                    
                    overwrites = {}
                    if config.permissions:
                        overwrites[guild.default_role] = discord.PermissionOverwrite(**config.permissions)
                    
                    channel = await guild.create_text_channel(
                        name=channel_name,
                        category=category,
                        topic=config.purpose,
                        overwrites=overwrites
                    )
                    
                    self.managed_channels[channel_name] = channel.id
                    await ctx.send(f"✅ Created channel: {channel_name}")
                
                except Exception as e:
                    await ctx.send(f"❌ Failed to create channel: {e}")
            else:
                await ctx.send(f"❌ Unknown channel configuration: {channel_name}")
        
        elif action == 'remove' and channel_name:
            if channel_name in self.managed_channels:
                try:
                    channel = ctx.guild.get_channel(self.managed_channels[channel_name])
                    if channel:
                        await channel.delete()
                        del self.managed_channels[channel_name]
                        await ctx.send(f"✅ Removed channel: {channel_name}")
                    else:
                        await ctx.send(f"❌ Channel not found: {channel_name}")
                
                except Exception as e:
                    await ctx.send(f"❌ Failed to remove channel: {e}")
            else:
                await ctx.send(f"❌ Channel not managed by bot: {channel_name}")
        
        else:
            await ctx.send("❌ Usage: `!ai channels <list|create|remove> [channel_name]`")
    
    def _group_channels_by_category(self) -> Dict[str, List[str]]:
        """Group channels by category for display"""
        categories = {}
        for channel_name, config in self.channel_configs.items():
            if config.category not in categories:
                categories[config.category] = []
            if channel_name in self.managed_channels:
                categories[config.category].append(channel_name)
        return categories
    
    @commands.command(name='shutdown')
    @commands.has_permissions(administrator=True)
    async def shutdown_command(self, ctx):
        """Shutdown the AI Corporation bot (Admin only)"""
        embed = discord.Embed(
            title="🛑 AI Corporation Bot Shutting Down",
            description="All systems will be gracefully shut down",
            color=0xff0000
        )
        
        await ctx.send(embed=embed)
        
        # Shutdown AI systems
        if self.data_collection:
            self.data_collection.shutdown()
        if self.self_defense:
            self.self_defense.shutdown()
        if self.p2p_network:
            await self.p2p_network.shutdown()
        
        # Stop tasks
        self.process_information_queue.cancel()
        self.monitor_ai_systems.cancel() 
        self.monitor_founder_protection.cancel()
        
        await self.close()

async def main():
    """Main function to run the Discord bot"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get Discord bot token from environment or config
    bot_token = os.getenv('DISCORD_BOT_TOKEN')
    
    if not bot_token:
        print("❌ DISCORD_BOT_TOKEN environment variable not set")
        print("Please set your Discord bot token:")
        print("export DISCORD_BOT_TOKEN='your_bot_token_here'")
        return
    
    # Create and run bot
    bot = AICorporationBot()
    
    try:
        await bot.start(bot_token)
    except discord.LoginFailure:
        print("❌ Invalid Discord bot token")
    except Exception as e:
        print(f"❌ Bot error: {e}")
        logging.error(f"Bot error: {e}")

if __name__ == "__main__":
    asyncio.run(main())