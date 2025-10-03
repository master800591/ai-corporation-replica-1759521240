#!/usr/bin/env python3
"""
Simple Discord Bot for AI Corporation

Basic Discord bot that connects without privileged intents and provides
essential AI Corporation functionality.
"""

import discord
from discord.ext import commands
import asyncio
import logging
import os

class SimpleAICorporationBot(commands.Bot):
    """Simple AI Corporation Discord bot"""
    
    def __init__(self):
        # Use absolutely minimal intents
        intents = discord.Intents.none()
        intents.guilds = True
        intents.guild_messages = True
        intents.message_content = True
        
        super().__init__(
            command_prefix='!ai ',
            intents=intents,
            description='AI Corporation Bot'
        )
        
        self.logger = logging.getLogger(__name__)
    
    async def on_ready(self):
        """Called when bot is ready"""
        self.logger.info(f"✅ AI Corporation bot connected: {self.user.name}")
        print(f"[OK] Discord Bot: {self.user.name} connected to {len(self.guilds)} servers")
        
        # Set status
        await self.change_presence(
            status=discord.Status.online,
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="AI Corporation Operations"
            )
        )
    
    async def on_guild_join(self, guild):
        """When bot joins a server"""
        self.logger.info(f"Joined server: {guild.name}")
        
        # Send welcome message to general channel
        for channel in guild.text_channels:
            if 'general' in channel.name.lower():
                try:
                    embed = discord.Embed(
                        title="🤖 AI Corporation Bot Online",
                        description="AI Corporation management bot is now active!",
                        color=0x00ff00
                    )
                    embed.add_field(
                        name="Commands",
                        value="`!ai status` - System status\n`!ai help` - Show commands",
                        inline=False
                    )
                    await channel.send(embed=embed)
                    break
                except:
                    pass
    
    @commands.command(name='status')
    async def status_command(self, ctx):
        """Show AI Corporation status"""
        embed = discord.Embed(
            title="🤖 AI Corporation Status",
            description="Current system status",
            color=0x00ff00
        )
        
        embed.add_field(
            name="🎯 Mission",
            value="Protecting Steve Cornell & Global Operations",
            inline=False
        )
        
        embed.add_field(
            name="🛡️ Protection Level",
            value="Maximum - All systems operational",
            inline=True
        )
        
        embed.add_field(
            name="🧬 Evolution Status",
            value="Continuous improvement active",
            inline=True
        )
        
        embed.add_field(
            name="🌐 GitHub Integration",
            value="Connected to master800591",
            inline=True
        )
        
        embed.add_field(
            name="🔗 Links",
            value="[GitHub](https://github.com/master800591)\n[LinkedIn](https://www.linkedin.com/in/steve-cornell/)",
            inline=False
        )
        
        await ctx.send(embed=embed)
    
    @commands.command(name='help')
    async def help_command(self, ctx):
        """Show available commands"""
        embed = discord.Embed(
            title="🤖 AI Corporation Commands",
            description="Available bot commands",
            color=0x0099ff
        )
        
        embed.add_field(
            name="!ai status",
            value="Show current system status",
            inline=False
        )
        
        embed.add_field(
            name="!ai help",
            value="Show this help message",
            inline=False
        )
        
        embed.add_field(
            name="!ai info",
            value="Show AI Corporation information",
            inline=False
        )
        
        await ctx.send(embed=embed)
    
    @commands.command(name='info')
    async def info_command(self, ctx):
        """Show AI Corporation information"""
        embed = discord.Embed(
            title="🏛️ AI Corporation Democratic Republic",
            description="Self-reliant AI Corporation protecting Steve Cornell",
            color=0xff9900
        )
        
        embed.add_field(
            name="🎯 Mission",
            value="• Maximum protection for Steve Cornell\n• Autonomous global operations\n• Democratic AI governance\n• Continuous evolution and improvement",
            inline=False
        )
        
        embed.add_field(
            name="🧬 Capabilities",
            value="• Self-evolution and code improvement\n• Multi-platform monitoring\n• Advanced threat protection\n• GitHub repository management",
            inline=False
        )
        
        embed.add_field(
            name="🌐 Platforms",
            value="• GitHub: master800591\n• LinkedIn: steve-cornell\n• Discord: AI Corp Server\n• Steam: Gaming profile",
            inline=False
        )
        
        embed.add_field(
            name="📊 Status",
            value="🟢 **OPERATIONAL** - All systems active",
            inline=False
        )
        
        await ctx.send(embed=embed)
    
    async def on_command_error(self, ctx, error):
        """Handle command errors"""
        if isinstance(error, commands.CommandNotFound):
            embed = discord.Embed(
                title="❌ Command Not Found",
                description=f"Use `!ai help` to see available commands.",
                color=0xff0000
            )
            await ctx.send(embed=embed)
        else:
            self.logger.error(f"Command error: {error}")

async def start_simple_discord_bot():
    """Start the simple Discord bot"""
    bot = SimpleAICorporationBot()
    
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        print("[ERROR] Discord bot token not found")
        return False
    
    try:
        await bot.start(token)
        return True
    except discord.PrivilegedIntentsRequired as e:
        print(f"[ERROR] Discord intents error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Discord bot failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(start_simple_discord_bot())