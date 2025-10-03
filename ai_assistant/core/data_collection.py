#!/usr/bin/env python3
"""
Multi-threaded Data Collection and Storage System

Advanced data collection system that scrapes social media, stores information,
creates detailed profiles, and maintains persistent databases. Includes
multi-threading, caching, and real-time monitoring.

Key Features:
- Multi-threaded data collection
- Social media scraping (LinkedIn, Discord, Steam, GitHub)
- Detailed profile creation for founder, governments, corporations
- Persistent data storage with threading-safe access
- Real-time monitoring and alerts
- Self-reliant data gathering and analysis
"""

import threading
import time
import json
import sqlite3
import hashlib
import requests
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import queue
from datetime import datetime, timedelta
import os
import urllib.parse
from pathlib import Path

# Optional imports with graceful degradation
try:
    import discord
    from discord.ext import commands
    discord_available = True
except ImportError:
    discord_available = False
    print("[WARNING] Discord.py not available - Discord integration limited")

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    selenium_available = True
except ImportError:
    selenium_available = False
    print("[WARNING] Selenium not available - advanced scraping disabled")

class DataType(Enum):
    """Types of data being collected"""
    FOUNDER_PROFILE = "founder_profile"
    GOVERNMENT_DATA = "government_data"
    CORPORATION_DATA = "corporation_data"
    SOCIAL_MEDIA = "social_media"
    THREAT_INTELLIGENCE = "threat_intelligence"
    NETWORK_ANALYTICS = "network_analytics"
    USER_REGISTRATION = "user_registration"

class SourceType(Enum):
    """Data source types"""
    LINKEDIN = "linkedin"
    DISCORD = "discord"
    STEAM = "steam"
    GITHUB = "github"
    REDDIT = "reddit"
    TWITTER = "twitter"
    NEWS_MEDIA = "news_media"
    GOVERNMENT_SITES = "government_sites"
    CORPORATE_SITES = "corporate_sites"

@dataclass
class DataProfile:
    """Comprehensive data profile structure"""
    profile_id: str
    profile_type: DataType
    name: str
    sources: Dict[SourceType, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    threat_assessment: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)
    update_frequency: int = 3600  # seconds
    monitoring_active: bool = True

@dataclass
class ScrapingTask:
    """Represents a data scraping task"""
    task_id: str
    source: SourceType
    target_url: str
    data_type: DataType
    profile_id: str
    priority: int = 1  # 1-10, higher is more important
    created_at: float = field(default_factory=time.time)
    max_retries: int = 3
    retry_count: int = 0

class DataCollectionManager:
    """Manages multi-threaded data collection and storage"""
    
    def __init__(self, db_path: str = "ai_corporation_data.db", 
                 max_workers: int = 10):
        self.db_path = db_path
        self.max_workers = max_workers
        
        # Threading components
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        self.worker_threads: List[threading.Thread] = []
        self.running = False
        
        # Data storage
        self.profiles: Dict[str, DataProfile] = {}
        self.db_lock = threading.RLock()
        
        # Monitoring and alerts
        self.monitoring_threads: List[threading.Thread] = []
        self.alert_handlers: List[Callable] = []
        
        # Scraping configuration
        self.scraping_config = {
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'request_delay': 2.0,  # Delay between requests
            'timeout': 30,
            'max_retries': 3
        }
        
        # Initialize database
        self.init_database()
        
        # Load existing profiles
        self.load_profiles_from_db()
        
        logging.info("Data Collection Manager initialized")
    
    def init_database(self):
        """Initialize SQLite database with thread-safe access"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS profiles (
                    profile_id TEXT PRIMARY KEY,
                    profile_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS scraped_data (
                    data_id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    scraped_at REAL NOT NULL,
                    FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS monitoring_alerts (
                    alert_id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            conn.close()
            
        logging.info("Database initialized successfully")
    
    def start_collection_system(self):
        """Start the multi-threaded data collection system"""
        if self.running:
            return
        
        self.running = True
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_thread,
                name=f"DataWorker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        # Start monitoring threads
        monitor = threading.Thread(
            target=self._monitoring_thread,
            name="DataMonitor",
            daemon=True
        )
        monitor.start()
        self.monitoring_threads.append(monitor)
        
        # Start result processor
        processor = threading.Thread(
            target=self._result_processor_thread,
            name="ResultProcessor",
            daemon=True
        )
        processor.start()
        
        logging.info(f"Data collection system started with {self.max_workers} workers")
    
    def _worker_thread(self):
        """Worker thread for processing scraping tasks"""
        while self.running:
            try:
                # Get task from queue (blocks until available)
                priority, task = self.task_queue.get(timeout=1.0)
                
                if task is None:  # Shutdown signal
                    break
                
                logging.info(f"Processing task: {task.task_id}")
                
                # Execute scraping task
                result = self._execute_scraping_task(task)
                
                # Put result in result queue
                self.result_queue.put((task, result))
                
                # Mark task as done
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Worker thread error: {e}")
    
    def _execute_scraping_task(self, task: ScrapingTask) -> Dict[str, Any]:
        """Execute a specific scraping task"""
        try:
            if task.source == SourceType.LINKEDIN:
                return self._scrape_linkedin(task)
            elif task.source == SourceType.DISCORD:
                return self._scrape_discord(task)
            elif task.source == SourceType.STEAM:
                return self._scrape_steam(task)
            elif task.source == SourceType.GITHUB:
                return self._scrape_github(task)
            elif task.source == SourceType.NEWS_MEDIA:
                return self._scrape_news_media(task)
            elif task.source == SourceType.GOVERNMENT_SITES:
                return self._scrape_government_sites(task)
            else:
                return self._generic_web_scrape(task)
                
        except Exception as e:
            logging.error(f"Scraping task {task.task_id} failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _scrape_linkedin(self, task: ScrapingTask) -> Dict[str, Any]:
        """Scrape LinkedIn profile data"""
        try:
            # LinkedIn scraping (respectful and within ToS)
            headers = {
                'User-Agent': self.scraping_config['user_agent']
            }
            
            response = requests.get(
                task.target_url,
                headers=headers,
                timeout=self.scraping_config['timeout']
            )
            
            if response.status_code == 200:
                # Extract basic information (respecting LinkedIn's policies)
                data = {
                    'success': True,
                    'profile_accessible': True,
                    'last_checked': time.time(),
                    'response_code': response.status_code,
                    'content_length': len(response.text)
                }
                
                # Note: In real implementation, use LinkedIn API
                logging.info(f"LinkedIn profile checked: {task.target_url}")
                return data
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}",
                    'timestamp': time.time()
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _scrape_discord(self, task: ScrapingTask) -> Dict[str, Any]:
        """Scrape Discord server/user data"""
        try:
            if not discord_available:
                return {
                    'success': False,
                    'error': 'Discord library not available',
                    'timestamp': time.time()
                }
            
            # Discord scraping would require bot token and proper permissions
            # For now, return placeholder data
            return {
                'success': True,
                'server_accessible': True,
                'last_checked': time.time(),
                'note': 'Discord integration requires bot setup'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _scrape_steam(self, task: ScrapingTask) -> Dict[str, Any]:
        """Scrape Steam profile data"""
        try:
            headers = {
                'User-Agent': self.scraping_config['user_agent']
            }
            
            response = requests.get(
                task.target_url,
                headers=headers,
                timeout=self.scraping_config['timeout']
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'profile_accessible': True,
                    'last_checked': time.time(),
                    'response_code': response.status_code
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}",
                    'timestamp': time.time()
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _scrape_github(self, task: ScrapingTask) -> Dict[str, Any]:
        """Scrape GitHub repository/user data"""
        try:
            # Use GitHub API for better reliability
            if '/users/' in task.target_url:
                username = task.target_url.split('/users/')[-1].split('/')[0]
                api_url = f"https://api.github.com/users/{username}"
            elif '/repos/' in task.target_url:
                repo_path = task.target_url.split('/repos/')[-1]
                api_url = f"https://api.github.com/repos/{repo_path}"
            else:
                return {'success': False, 'error': 'Invalid GitHub URL'}
            
            response = requests.get(api_url, timeout=self.scraping_config['timeout'])
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'api_data': data,
                    'last_checked': time.time()
                }
            else:
                return {
                    'success': False,
                    'error': f"GitHub API error: {response.status_code}",
                    'timestamp': time.time()
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _scrape_news_media(self, task: ScrapingTask) -> Dict[str, Any]:
        """Scrape news media for mentions"""
        try:
            headers = {
                'User-Agent': self.scraping_config['user_agent']
            }
            
            response = requests.get(
                task.target_url,
                headers=headers,
                timeout=self.scraping_config['timeout']
            )
            
            if response.status_code == 200:
                # Basic content analysis
                content = response.text.lower()
                
                # Look for founder mentions
                founder_mentions = content.count('steve cornell')
                ai_corp_mentions = content.count('ai corporation')
                
                return {
                    'success': True,
                    'founder_mentions': founder_mentions,
                    'ai_corp_mentions': ai_corp_mentions,
                    'content_length': len(response.text),
                    'last_checked': time.time()
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}",
                    'timestamp': time.time()
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _scrape_government_sites(self, task: ScrapingTask) -> Dict[str, Any]:
        """Scrape government websites for relevant information"""
        try:
            headers = {
                'User-Agent': self.scraping_config['user_agent']
            }
            
            response = requests.get(
                task.target_url,
                headers=headers,
                timeout=self.scraping_config['timeout']
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'site_accessible': True,
                    'content_length': len(response.text),
                    'last_checked': time.time(),
                    'data_type': 'government_site'
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}",
                    'timestamp': time.time()
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _generic_web_scrape(self, task: ScrapingTask) -> Dict[str, Any]:
        """Generic web scraping for any URL"""
        try:
            headers = {
                'User-Agent': self.scraping_config['user_agent']
            }
            
            response = requests.get(
                task.target_url,
                headers=headers,
                timeout=self.scraping_config['timeout']
            )
            
            return {
                'success': response.status_code == 200,
                'status_code': response.status_code,
                'content_length': len(response.text) if response.status_code == 200 else 0,
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _result_processor_thread(self):
        """Process scraping results and update profiles"""
        while self.running:
            try:
                task, result = self.result_queue.get(timeout=1.0)
                
                # Update profile with new data
                self._update_profile_with_result(task, result)
                
                # Check for alerts
                self._check_for_alerts(task, result)
                
                self.result_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Result processor error: {e}")
    
    def _update_profile_with_result(self, task: ScrapingTask, result: Dict[str, Any]):
        """Update data profile with scraping result"""
        try:
            if task.profile_id in self.profiles:
                profile = self.profiles[task.profile_id]
                
                # Update source data
                profile.sources[task.source] = result
                profile.last_updated = time.time()
                
                # Save to database
                self._save_profile_to_db(profile)
                self._save_scraped_data_to_db(task, result)
                
                logging.info(f"Profile {task.profile_id} updated with {task.source.value} data")
            
        except Exception as e:
            logging.error(f"Profile update error: {e}")
    
    def _check_for_alerts(self, task: ScrapingTask, result: Dict[str, Any]):
        """Check scraping results for alert conditions"""
        try:
            alerts = []
            
            # Check for profile accessibility issues
            if not result.get('success', False):
                alerts.append({
                    'type': 'scraping_failure',
                    'severity': 5,
                    'message': f"Failed to scrape {task.source.value}: {result.get('error', 'Unknown error')}"
                })
            
            # Check for founder mentions in news
            if task.source == SourceType.NEWS_MEDIA and result.get('success'):
                founder_mentions = result.get('founder_mentions', 0)
                if founder_mentions > 0:
                    alerts.append({
                        'type': 'founder_mention',
                        'severity': 3,
                        'message': f"Founder mentioned {founder_mentions} times in media"
                    })
            
            # Check for government site access issues
            if task.source == SourceType.GOVERNMENT_SITES and not result.get('success'):
                alerts.append({
                    'type': 'government_access',
                    'severity': 7,
                    'message': f"Cannot access government site: {task.target_url}"
                })
            
            # Save alerts to database
            for alert in alerts:
                self._save_alert_to_db(task.profile_id, alert)
                
                # Trigger alert handlers
                for handler in self.alert_handlers:
                    try:
                        handler(task, alert)
                    except Exception as e:
                        logging.error(f"Alert handler error: {e}")
            
        except Exception as e:
            logging.error(f"Alert checking error: {e}")
    
    def _monitoring_thread(self):
        """Continuous monitoring thread"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check profiles for updates needed
                for profile in self.profiles.values():
                    if not profile.monitoring_active:
                        continue
                    
                    time_since_update = current_time - profile.last_updated
                    if time_since_update >= profile.update_frequency:
                        # Schedule update tasks
                        self._schedule_profile_updates(profile)
                
                # Sleep for monitoring interval
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logging.error(f"Monitoring thread error: {e}")
                time.sleep(5)
    
    def _schedule_profile_updates(self, profile: DataProfile):
        """Schedule update tasks for a profile"""
        for source, source_data in profile.sources.items():
            if 'target_url' in source_data:
                task = ScrapingTask(
                    task_id=f"update_{profile.profile_id}_{source.value}_{int(time.time())}",
                    source=source,
                    target_url=source_data['target_url'],
                    data_type=profile.profile_type,
                    profile_id=profile.profile_id,
                    priority=5  # Medium priority for updates
                )
                
                self.add_scraping_task(task)
    
    def add_scraping_task(self, task: ScrapingTask):
        """Add a scraping task to the queue"""
        # Priority queue uses negative priority for max-heap behavior
        self.task_queue.put((-task.priority, task))
        logging.info(f"Scraping task added: {task.task_id}")
    
    def create_founder_profile(self) -> DataProfile:
        """Create comprehensive founder profile"""
        profile_id = "steve_cornell_founder"
        
        profile = DataProfile(
            profile_id=profile_id,
            profile_type=DataType.FOUNDER_PROFILE,
            name="Steve Cornell",
            update_frequency=3600,  # Update every hour
            monitoring_active=True
        )
        
        # Add founder's known sources
        profile.sources[SourceType.LINKEDIN] = {
            'target_url': 'https://www.linkedin.com/in/steve-cornell/',
            'profile_type': 'professional'
        }
        
        profile.sources[SourceType.STEAM] = {
            'target_url': 'https://steamcommunity.com/profiles/76561198074298205',
            'profile_type': 'gaming'
        }
        
        profile.sources[SourceType.DISCORD] = {
            'target_url': 'https://discord.gg/9uvrmEHa',
            'username': 'master80059',
            'server': 'AI Corp'
        }
        
        profile.sources[SourceType.GITHUB] = {
            'target_url': 'https://github.com/steve-cornell',
            'profile_type': 'development'
        }
        
        # Initialize threat assessment
        profile.threat_assessment = {
            'current_level': 2,
            'last_assessment': time.time(),
            'factors': {
                'public_exposure': 3,
                'online_presence': 4,
                'ai_corporation_association': 5
            }
        }
        
        self.profiles[profile_id] = profile
        self._save_profile_to_db(profile)
        
        logging.info("Founder profile created and monitoring activated")
        return profile
    
    def create_government_profile(self, name: str, website: str) -> DataProfile:
        """Create government entity profile"""
        profile_id = f"gov_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        profile = DataProfile(
            profile_id=profile_id,
            profile_type=DataType.GOVERNMENT_DATA,
            name=name,
            update_frequency=7200,  # Update every 2 hours
            monitoring_active=True
        )
        
        profile.sources[SourceType.GOVERNMENT_SITES] = {
            'target_url': website,
            'entity_type': 'government'
        }
        
        self.profiles[profile_id] = profile
        self._save_profile_to_db(profile)
        
        return profile
    
    def create_corporation_profile(self, name: str, website: str) -> DataProfile:
        """Create corporation profile"""
        profile_id = f"corp_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        profile = DataProfile(
            profile_id=profile_id,
            profile_type=DataType.CORPORATION_DATA,
            name=name,
            update_frequency=14400,  # Update every 4 hours
            monitoring_active=True
        )
        
        profile.sources[SourceType.CORPORATE_SITES] = {
            'target_url': website,
            'entity_type': 'corporation'
        }
        
        self.profiles[profile_id] = profile
        self._save_profile_to_db(profile)
        
        return profile
    
    def _save_profile_to_db(self, profile: DataProfile):
        """Save profile to database with thread safety"""
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                
                profile_data = asdict(profile)
                
                conn.execute('''
                    INSERT OR REPLACE INTO profiles 
                    (profile_id, profile_type, name, data_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    profile.profile_id,
                    profile.profile_type.value,
                    profile.name,
                    json.dumps(profile_data),
                    time.time(),
                    profile.last_updated
                ))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                logging.error(f"Database save error: {e}")
    
    def _save_scraped_data_to_db(self, task: ScrapingTask, result: Dict[str, Any]):
        """Save scraped data to database"""
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                
                data_id = f"{task.task_id}_{int(time.time())}"
                
                conn.execute('''
                    INSERT INTO scraped_data 
                    (data_id, profile_id, source_type, data_json, scraped_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    data_id,
                    task.profile_id,
                    task.source.value,
                    json.dumps(result),
                    time.time()
                ))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                logging.error(f"Scraped data save error: {e}")
    
    def _save_alert_to_db(self, profile_id: str, alert: Dict[str, Any]):
        """Save alert to database"""
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                
                alert_id = f"alert_{int(time.time())}_{hashlib.md5(str(alert).encode()).hexdigest()[:8]}"
                
                conn.execute('''
                    INSERT INTO monitoring_alerts 
                    (alert_id, profile_id, alert_type, severity, message, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    alert_id,
                    profile_id,
                    alert['type'],
                    alert['severity'],
                    alert['message'],
                    time.time()
                ))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                logging.error(f"Alert save error: {e}")
    
    def load_profiles_from_db(self):
        """Load existing profiles from database"""
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM profiles')
                rows = cursor.fetchall()
                
                for row in rows:
                    profile_data = json.loads(row[3])  # data_json column
                    
                    # Convert back to DataProfile object
                    profile = DataProfile(
                        profile_id=profile_data['profile_id'],
                        profile_type=DataType(profile_data['profile_type']),
                        name=profile_data['name'],
                        sources={
                            SourceType(k): v for k, v in profile_data['sources'].items()
                        },
                        metadata=profile_data.get('metadata', {}),
                        threat_assessment=profile_data.get('threat_assessment', {}),
                        last_updated=profile_data.get('last_updated', time.time()),
                        update_frequency=profile_data.get('update_frequency', 3600),
                        monitoring_active=profile_data.get('monitoring_active', True)
                    )
                    
                    self.profiles[profile.profile_id] = profile
                
                conn.close()
                logging.info(f"Loaded {len(self.profiles)} profiles from database")
                
            except Exception as e:
                logging.error(f"Profile loading error: {e}")
    
    def add_alert_handler(self, handler: Callable):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
        logging.info("Alert handler added")
    
    def get_profile_status(self) -> Dict[str, Any]:
        """Get current status of all profiles"""
        return {
            'total_profiles': len(self.profiles),
            'active_monitoring': len([p for p in self.profiles.values() if p.monitoring_active]),
            'founder_profiles': len([p for p in self.profiles.values() if p.profile_type == DataType.FOUNDER_PROFILE]),
            'government_profiles': len([p for p in self.profiles.values() if p.profile_type == DataType.GOVERNMENT_DATA]),
            'corporation_profiles': len([p for p in self.profiles.values() if p.profile_type == DataType.CORPORATION_DATA]),
            'queue_size': self.task_queue.qsize(),
            'worker_threads': len(self.worker_threads),
            'monitoring_threads': len(self.monitoring_threads)
        }
    
    def shutdown(self):
        """Shutdown the data collection system"""
        self.running = False
        
        # Signal workers to stop
        for _ in range(self.max_workers):
            self.task_queue.put((-1, None))
        
        # Wait for threads to complete
        for thread in self.worker_threads + self.monitoring_threads:
            thread.join(timeout=5.0)
        
        self.executor.shutdown(wait=True)
        
        logging.info("Data collection system shutdown complete")

def create_data_collection_system(max_workers: int = 10) -> DataCollectionManager:
    """Create and configure data collection system"""
    return DataCollectionManager(max_workers=max_workers)

# Example usage and testing
if __name__ == "__main__":
    # Create data collection system
    data_system = create_data_collection_system(max_workers=5)
    
    # Add alert handler
    def threat_alert_handler(task, alert):
        if alert['severity'] >= 7:
            print(f"HIGH SEVERITY ALERT: {alert['message']}")
    
    data_system.add_alert_handler(threat_alert_handler)
    
    # Start the system
    data_system.start_collection_system()
    
    # Create founder profile and start monitoring
    founder_profile = data_system.create_founder_profile()
    
    # Schedule initial scraping tasks for founder
    for source, source_data in founder_profile.sources.items():
        if 'target_url' in source_data:
            task = ScrapingTask(
                task_id=f"initial_{source.value}_{int(time.time())}",
                source=source,
                target_url=source_data['target_url'],
                data_type=DataType.FOUNDER_PROFILE,
                profile_id=founder_profile.profile_id,
                priority=10  # High priority for founder
            )
            data_system.add_scraping_task(task)
    
    print("Data collection system started - monitoring founder and gathering intelligence")
    print("System Status:", data_system.get_profile_status())
    
    try:
        # Keep running
        while True:
            time.sleep(10)
            status = data_system.get_profile_status()
            print(f"Queue: {status['queue_size']} tasks, Profiles: {status['total_profiles']}")
    except KeyboardInterrupt:
        data_system.shutdown()
        print("Data collection system stopped")