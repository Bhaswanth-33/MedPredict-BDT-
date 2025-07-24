#!/usr/bin/env python3
"""
Hospital Data Heat Analysis with Apache Spark
=============================================

This script performs:
1. Data-centric heat analysis using Apache Spark.
2. Interactive visualizations with FIXED layouts and color scales.
3. Comprehensive system resource monitoring (CPU, Memory, Disk, Network).
4. Performance metrics visualization and reporting.
"""

import os
import sys
import time
import threading  # FIXED: Added missing import
import shutil
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict

# System monitoring imports
import psutil

# Spark imports
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Visualization imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

class SystemMonitor:
    """
    Comprehensive system resource monitoring class (GPU part removed).
    """
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = defaultdict(deque)
        self.start_time = None
        self.process = psutil.Process()
        
        self.system_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total
        }
        
        print(f"🖥️  System Monitor Initialized: {self.system_info['cpu_count']} CPU Cores, {self.system_info['memory_total'] / (1024**3):.2f} GB Memory")
    
    def start_monitoring(self):
        if self.monitoring: return
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("📊 System monitoring started")
    
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join()
        print("🛑 System monitoring stopped")
    
    def _monitor_loop(self):
        while self.monitoring:
            try:
                timestamp = time.time() - self.start_time
                self.metrics_history['timestamp'].append(timestamp)
                
                # CPU and Memory
                self.metrics_history['cpu_percent'].append(psutil.cpu_percent(interval=None))
                self.metrics_history['system_memory_percent'].append(psutil.virtual_memory().percent)
                self.metrics_history['process_memory_mb'].append(self.process.memory_info().rss / (1024**2))
                
                # Disk and Network I/O
                disk_io = psutil.disk_io_counters()
                net_io = psutil.net_io_counters()
                if disk_io:
                    self.metrics_history['disk_read_mb'].append(disk_io.read_bytes / (1024**2))
                    self.metrics_history['disk_write_mb'].append(disk_io.write_bytes / (1024**2))
                if net_io:
                    self.metrics_history['network_sent_mb'].append(net_io.bytes_sent / (1024**2))
                    self.metrics_history['network_recv_mb'].append(net_io.bytes_recv / (1024**2))

                for key in self.metrics_history:
                    if len(self.metrics_history[key]) > 1000:
                        self.metrics_history[key].popleft()
                time.sleep(self.sampling_interval)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.monitoring = False
            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")

    def generate_performance_report(self) -> Dict:
        if not self.metrics_history['timestamp']: return {}
        report = {
            'duration_seconds': self.metrics_history['timestamp'][-1] if self.metrics_history['timestamp'] else 0,
            'cpu_stats': {
                'max_percent': max(self.metrics_history['cpu_percent'], default=0),
                'avg_percent': np.mean(self.metrics_history['cpu_percent']) if self.metrics_history['cpu_percent'] else 0,
            },
            'memory_stats': {
                'peak_process_memory_mb': max(self.metrics_history['process_memory_mb'], default=0),
            },
            'disk_stats': {
                'total_read_mb': self.metrics_history['disk_read_mb'][-1] - self.metrics_history['disk_read_mb'][0] if len(self.metrics_history.get('disk_read_mb', [])) > 1 else 0,
                'total_write_mb': self.metrics_history['disk_write_mb'][-1] - self.metrics_history['disk_write_mb'][0] if len(self.metrics_history.get('disk_write_mb', [])) > 1 else 0,
            },
            'network_stats': {
                'total_sent_mb': self.metrics_history['network_sent_mb'][-1] - self.metrics_history['network_sent_mb'][0] if len(self.metrics_history.get('network_sent_mb', [])) > 1 else 0,
                'total_recv_mb': self.metrics_history['network_recv_mb'][-1] - self.metrics_history['network_recv_mb'][0] if len(self.metrics_history.get('network_recv_mb', [])) > 1 else 0,
            }
        }
        return report

class HospitalDataAnalyzer:
    """
    Performs batch analysis using Apache Spark.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.spark = None
        self.df = None
        self.monitor = SystemMonitor(sampling_interval=0.5)

    def _initialize_spark(self) -> SparkSession:
        print("⚡ Initializing Spark session...")
        self.monitor.start_monitoring()
        
        spark = SparkSession.builder \
            .appName("HospitalDataAnalysis") \
            .config("spark.sql.shuffle.partitions", "8") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.ui.showConsoleProgress", "false") \
            .getOrCreate()
            
        spark.sparkContext.setLogLevel("ERROR")
        print("✅ Spark Session initialized successfully.")
        return spark

    def load_and_prepare_data(self) -> DataFrame:
        print("\n--- 1. Starting BATCH Analysis ---")
        print(f"📊 Loading data from: {self.data_path}")
        if self.spark is None:
            self.spark = self._initialize_spark()

        schema = StructType([
            StructField("Name", StringType()), StructField("Age", IntegerType()), StructField("Gender", StringType()),
            StructField("Blood Type", StringType()), StructField("Medical Condition", StringType()),
            StructField("Date of Admission", StringType()), StructField("Doctor", StringType()),
            StructField("Hospital", StringType()), StructField("Insurance Provider", StringType()),
            StructField("Billing Amount", DoubleType()), StructField("Room Number", StringType()),
            StructField("Admission Type", StringType()), StructField("Discharge Date", StringType()),
            StructField("Medication", StringType()), StructField("Test Results", StringType()),
            StructField("Region", StringType()), StructField("Area_Type", StringType()),
            StructField("Emergency_Risk", StringType()), StructField("Insurance_Type", StringType()),
            StructField("Eligible_For_Financial_Aid", StringType()), StructField("Income_Bracket", StringType())
        ])
        
        self.df = self.spark.read.option("header", "true").schema(schema).csv(self.data_path)
        
        self.df = self.df.withColumnRenamed("Medical Condition", "Medical_Condition") \
                         .withColumnRenamed("Insurance Provider", "Insurance_Provider") \
                         .withColumnRenamed("Billing Amount", "Billing_Amount") \
                         .withColumn("Date_of_Admission", F.to_date(F.col("Date of Admission"), "yyyy-MM-dd")) \
                         .withColumn("Admission_Year", F.year(F.col("Date_of_Admission"))) \
                         .withColumn("Admission_Month", F.month(F.col("Date_of_Admission"))) \
                         .cache()
        
        record_count = self.df.count()
        print(f"✅ Data loaded and prepared: {record_count:,} records.")
        return self.df

    def generate_analysis_data(self) -> Dict[str, pd.DataFrame]:
        print("🔥 Generating data for all visualizations...")
        analysis_data = {}
        analysis_data['regional_distribution'] = self.df.groupBy("Region", "Area_Type").count().toPandas()
        analysis_data['condition_intensity'] = self.df.groupBy("Region", "Medical_Condition").count().toPandas()
        analysis_data['temporal_analysis'] = self.df.groupBy("Admission_Year", "Admission_Month").count().toPandas()
        analysis_data['emergency_risk'] = self.df.groupBy("Region", "Emergency_Risk").count().toPandas()
        analysis_data['hospital_volume'] = self.df.groupBy("Hospital").count().orderBy(F.desc("count")).limit(15).toPandas()
        analysis_data['avg_billing_by_insurance'] = self.df.groupBy("Insurance_Provider").agg(F.avg("Billing_Amount").alias("avg_billing")).orderBy(F.desc("avg_billing")).toPandas()
        
        print("✅ All visualization data generated.")
        return analysis_data

    def create_data_centric_dashboard(self, analysis_data: Dict[str, pd.DataFrame]):
        print("🎨 Creating expanded data-centric dashboard...")
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                '🔥 Patient Distribution by Region & Area Type',
                '🩺 Medical Conditions by Region (All Conditions)',
                '📅 Temporal Admission Heatmap (Year/Month)',
                '🚨 Emergency Risk Distribution by Region',
                '🏥 Top 15 Hospitals by Patient Volume',
                '💰 Average Billing Amount by Insurance Provider'
            ],
            vertical_spacing=0.20,  # Increased vertical spacing
            horizontal_spacing=0.25  # Increased horizontal spacing for color bars
        )

        # --- ROW 1 ---
        # Heatmap 1: Regional Distribution (Top-Left)
        regional_pivot = analysis_data['regional_distribution'].pivot(index='Region', columns='Area_Type', values='count').fillna(0)
        fig.add_trace(go.Heatmap(
            z=regional_pivot.values, 
            x=regional_pivot.columns, 
            y=regional_pivot.index,
            colorscale='Viridis', 
            colorbar=dict(
                title=dict(text="Patients", side="right"),
                thickness=15,
                len=0.25, 
                y=0.85,  # Top row position
                x=0.42,  # Left side, away from heatmap
                xanchor="left"
            ),
            hovertemplate='Region: %{y}<br>Area: %{x}<br>Count: %{z}<extra></extra>',
            name="Regional"
        ), row=1, col=1)

        # Heatmap 2: Medical Conditions (Top-Right)
        condition_pivot = analysis_data['condition_intensity'].pivot_table(index='Medical_Condition', columns='Region', values='count', fill_value=0)
        fig.add_trace(go.Heatmap(
            z=condition_pivot.values, 
            x=condition_pivot.columns, 
            y=condition_pivot.index,
            colorscale='Plasma', 
            colorbar=dict(
                title=dict(text="Cases", side="right"),
                thickness=15,
                len=0.25, 
                y=0.85,  # Top row position
                x=1.02,  # Right side, outside the plot area
                xanchor="left"
            ),
            hovertemplate='Condition: %{y}<br>Region: %{x}<br>Cases: %{z}<extra></extra>',
            name="Conditions"
        ), row=1, col=2)

        # --- ROW 2 ---
        # Heatmap 3: Temporal Analysis (Middle-Left)
        temporal_pivot = analysis_data['temporal_analysis'].pivot(index='Admission_Month', columns='Admission_Year', values='count').fillna(0)
        fig.add_trace(go.Heatmap(
            z=temporal_pivot.values, 
            x=temporal_pivot.columns, 
            y=temporal_pivot.index,
            colorscale='Cividis', 
            colorbar=dict(
                title=dict(text="Admissions", side="right"),
                thickness=15,
                len=0.25, 
                y=0.52,  # Middle row position
                x=0.42,  # Left side, away from heatmap
                xanchor="left"
            ),
            hovertemplate='Year: %{x}<br>Month: %{y}<br>Admissions: %{z}<extra></extra>',
            name="Temporal"
        ), row=2, col=1)

        # Heatmap 4: Emergency Risk (Middle-Right)
        risk_pivot = analysis_data['emergency_risk'].pivot_table(index="Emergency_Risk", columns="Region", values="count", fill_value=0)
        fig.add_trace(go.Heatmap(
            z=risk_pivot.values, 
            x=risk_pivot.columns, 
            y=risk_pivot.index,
            colorscale='Reds', 
            colorbar=dict(
                title=dict(text="Risk Cases", side="right"),
                thickness=15,
                len=0.25, 
                y=0.52,  # Middle row position
                x=1.02,  # Right side, outside the plot area
                xanchor="left"
            ),
            hovertemplate='Risk: %{y}<br>Region: %{x}<br>Cases: %{z}<extra></extra>',
            name="Emergency"
        ), row=2, col=2)

        # --- ROW 3 (Bar Charts - No color bars needed) ---
        hospital_df = analysis_data['hospital_volume']
        fig.add_trace(go.Bar(
            x=hospital_df['Hospital'], 
            y=hospital_df['count'],
            marker_color='skyblue', 
            text=hospital_df['count'], 
            textposition='outside',
            name="Hospital Volume"
        ), row=3, col=1)

        billing_df = analysis_data['avg_billing_by_insurance']
        fig.add_trace(go.Bar(
            x=billing_df['Insurance_Provider'], 
            y=billing_df['avg_billing'],
            marker_color='lightcoral', 
            text=billing_df['avg_billing'].apply(lambda x: f'${x:,.0f}'), 
            textposition='outside',
            name="Billing"
        ), row=3, col=2)
        
        fig.update_layout(
            height=2000,  # Increased height for better spacing
            width=2000,   # Increased width for color bars
            title_text="🏥 Hospital Data-Centric Analysis Dashboard",
            title_x=0.5, 
            showlegend=False,
            margin=dict(l=120, r=200, t=120, b=120)  # Increased right margin for color bars
        )
        
        # Update x-axis for better readability
        fig.update_xaxes(tickangle=45, row=3, col=1)
        fig.update_xaxes(tickangle=45, row=3, col=2)
        
        filename = f"hospital_data_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(filename)
        print(f"✅ Data-centric visualizations saved to '{filename}'")

    def create_performance_dashboard(self):
        print("\n--- 2. Generating Performance Report ---")
        report = self.monitor.generate_performance_report()
        if not report:
            print("⚠️ No performance data to report.")
            return

        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'CPU Usage Over Time (%)', 
                'System Memory Usage (%)',
                'Process Memory Usage (MB)',
                'CPU Load Distribution',
                'Disk I/O (Cumulative MB)',
                'Network I/O (Cumulative MB)',
                'Resource Usage Intensity Heatmap',
                'Key Performance Indicators'
            ],
            vertical_spacing=0.18,  # Increased vertical spacing
            horizontal_spacing=0.25,  # Increased horizontal spacing
            specs=[[{}, {}], [{}, {}], [{}, {}], [{}, {"type": "table"}]]
        )
        
        monitoring_df = pd.DataFrame({k: list(v) for k, v in self.monitor.metrics_history.items()})
        timestamps = monitoring_df['timestamp']

        # Row 1: Line charts (no color bars)
        fig.add_trace(go.Scatter(
            x=timestamps, 
            y=monitoring_df['cpu_percent'], 
            mode='lines', 
            name='CPU %', 
            line=dict(color='red')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=timestamps, 
            y=monitoring_df['system_memory_percent'], 
            mode='lines', 
            name='System Mem %', 
            line=dict(color='green')
        ), row=1, col=2)
        
        # Row 2: Process memory and CPU distribution
        fig.add_trace(go.Scatter(
            x=timestamps, 
            y=monitoring_df['process_memory_mb'], 
            mode='lines', 
            name='Process Mem MB', 
            line=dict(color='blue')
        ), row=2, col=1)
        
        fig.add_trace(go.Histogram(
            x=monitoring_df['cpu_percent'], 
            nbinsx=20, 
            name='CPU Bins', 
            marker_color='orange'
        ), row=2, col=2)
        
        # Row 3: Disk and Network I/O
        if 'disk_read_mb' in monitoring_df.columns:
            fig.add_trace(go.Scatter(
                x=timestamps, 
                y=monitoring_df['disk_read_mb'], 
                mode='lines', 
                name='Disk Read', 
                line=dict(color='purple')
            ), row=3, col=1)
        if 'disk_write_mb' in monitoring_df.columns:
            fig.add_trace(go.Scatter(
                x=timestamps, 
                y=monitoring_df['disk_write_mb'], 
                mode='lines', 
                name='Disk Write', 
                line=dict(color='brown')
            ), row=3, col=1)
        if 'network_sent_mb' in monitoring_df.columns:
            fig.add_trace(go.Scatter(
                x=timestamps, 
                y=monitoring_df['network_sent_mb'], 
                mode='lines', 
                name='Net Sent', 
                line=dict(color='cyan')
            ), row=3, col=2)
        if 'network_recv_mb' in monitoring_df.columns:
            fig.add_trace(go.Scatter(
                x=timestamps, 
                y=monitoring_df['network_recv_mb'], 
                mode='lines', 
                name='Net Recv', 
                line=dict(color='magenta')
            ), row=3, col=2)

        # Row 4: Resource heatmap with properly positioned color bar
        resource_cols = ['cpu_percent', 'system_memory_percent', 'process_memory_mb']
        heatmap_data = []
        for col in resource_cols:
            if col in monitoring_df.columns and not monitoring_df[col].empty:
                values = monitoring_df[col].values
                normalized = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-9)
                heatmap_data.append(normalized)
        
        if heatmap_data:
            fig.add_trace(go.Heatmap(
                z=heatmap_data, 
                y=resource_cols, 
                colorscale='Cividis',
                colorbar=dict(
                    title=dict(text="Intensity", side="right"),
                    thickness=15,
                    len=0.3, 
                    y=0.15,  # Bottom row position
                    x=0.42,  # Left side, away from heatmap
                    xanchor="left"
                )
            ), row=4, col=1)

        # Performance metrics table
        cpu_stats = report.get('cpu_stats', {})
        mem_stats = report.get('memory_stats', {})
        disk_stats = report.get('disk_stats', {})
        fig.add_trace(go.Table(
            header=dict(values=['Metric', 'Value'], fill_color='paleturquoise', align='left'),
            cells=dict(values=[
                ['Duration (s)', 'Max CPU %', 'Avg CPU %', 'Peak Proc Mem (MB)', 'Total Disk Read (MB)', 'Total Disk Write (MB)'],
                [f"{report['duration_seconds']:.1f}", f"{cpu_stats['max_percent']:.1f}", f"{cpu_stats['avg_percent']:.1f}", f"{mem_stats['peak_process_memory_mb']:.1f}", f"{disk_stats['total_read_mb']:.2f}", f"{disk_stats['total_write_mb']:.2f}"]
            ], fill_color='lavender', align='left')
        ), row=4, col=2)

        fig.update_layout(
            height=1800,  # Increased height
            width=1800,   # Increased width
            title_text="📊 System Performance Monitoring Dashboard",
            title_x=0.5, 
            showlegend=True,
            margin=dict(l=100, r=200, t=120, b=100)  # Increased right margin
        )
        
        filename = f"system_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(filename)
        print(f"✅ Performance report saved to '{filename}'")

    def _print_summary(self):
        print("\n" + "="*50)
        print("🏁 ANALYSIS SUMMARY 🏁")
        print("="*50)
        report = self.monitor.generate_performance_report()
        if report:
            print(f"⏱️  Total Duration: {report.get('duration_seconds', 0):.2f} seconds")
            cpu_stats = report.get('cpu_stats', {})
            print(f"🔥 Max CPU Usage: {cpu_stats.get('max_percent', 0):.1f}%")
            mem_stats = report.get('memory_stats', {})
            print(f"🧠 Peak Process Memory: {mem_stats.get('peak_process_memory_mb', 0):.1f} MB")
        else:
            print("No performance data collected.")
        print("="*50)
    
    def _cleanup(self):
        print("\n🔄 Cleaning up resources...")
        self.monitor.stop_monitoring()
        if self.spark:
            self.spark.stop()
            print("✅ Spark session stopped.")

    def run_analysis_pipeline(self):
        """Executes the full analysis pipeline from start to finish."""
        try:
            self.load_and_prepare_data()
            analysis_data = self.generate_analysis_data()
            self.create_data_centric_dashboard(analysis_data)
            self.create_performance_dashboard()
            self._print_summary()
        except Exception as e:
            print(f"\n❌ An error occurred during the analysis pipeline: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

def main():
    DATASET_PATH = "data/enriched_hospital_data.csv"
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERROR: Dataset not found at '{DATASET_PATH}'")
        print("Please ensure the CV file is in the correct location.")
        sys.exit(1)

    analyzer = HospitalDataAnalyzer(data_path=DATASET_PATH)
    analyzer.run_analysis_pipeline()

if __name__ == "__main__":
    main()