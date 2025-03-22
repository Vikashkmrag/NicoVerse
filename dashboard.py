"""
Dashboard for Document Retrieval Application

This module provides a Streamlit dashboard for visualizing logs and metrics
from the Document Retrieval Application.
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import glob
from collections import Counter, defaultdict
from modules.utils.logger import get_logger
from modules.utils.temp_message import show_temp_message

# Constants
LOG_DIR = './logs'
JSON_LOG_FILE = os.path.join(LOG_DIR, 'app.json.log')

def load_logs(days=7):
    """
    Load logs from the JSON log file
    
    Args:
        days (int): Number of days of logs to load
    
    Returns:
        list: List of log entries as dictionaries
    """
    logs = []
    
    # Check if log file exists
    if not os.path.exists(JSON_LOG_FILE):
        return logs
    
    # Calculate cutoff date
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    try:
        with open(JSON_LOG_FILE, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    
                    # Parse timestamp
                    timestamp = datetime.fromisoformat(log_entry['timestamp'].rstrip('Z'))
                    
                    # Skip entries older than cutoff date
                    if timestamp < cutoff_date:
                        continue
                    
                    # Add parsed timestamp
                    log_entry['parsed_timestamp'] = timestamp
                    
                    logs.append(log_entry)
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
    except Exception:
        pass
    
    return logs

def create_dashboard():
    """Create and display the dashboard"""
    st.title("Document Retrieval App Dashboard")
    
    # Sidebar for filtering
    st.sidebar.header("Filters")
    
    # Date range filter
    days_options = [1, 3, 7, 14, 30]
    selected_days = st.sidebar.selectbox(
        "Show data for last N days:",
        days_options,
        index=2  # Default to 7 days
    )
    
    # Load logs
    logs = load_logs(days=selected_days)
    
    if not logs:
        show_temp_message("No logs found. Start using the application to generate logs.", type="warning")
        return
    
    # Convert logs to DataFrame for easier analysis
    df = pd.DataFrame(logs)
    
    # Event type filter
    if 'event_type' in df.columns:
        event_types = ['All'] + sorted(df['event_type'].unique().tolist())
        selected_event_type = st.sidebar.selectbox(
            "Filter by event type:",
            event_types
        )
        
        if selected_event_type != 'All':
            df = df[df['event_type'] == selected_event_type]
    
    # Component filter
    if 'component' in df.columns:
        components = ['All'] + sorted(df['component'].unique().tolist())
        selected_component = st.sidebar.selectbox(
            "Filter by component:",
            components
        )
        
        if selected_component != 'All':
            df = df[df['component'] == selected_component]
    
    # Overview metrics
    st.header("Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Events", len(df))
    
    with col2:
        if 'level' in df.columns:
            error_count = len(df[df['level'].isin(['error', 'critical'])])
            st.metric("Errors", error_count)
    
    with col3:
        if 'event_type' in df.columns and 'model_usage' in df['event_type'].values:
            model_usage_count = len(df[df['event_type'] == 'model_usage'])
            st.metric("Model Calls", model_usage_count)
    
    with col4:
        if 'event_type' in df.columns and 'document_processing' in df['event_type'].values:
            doc_processing_count = len(df[df['event_type'] == 'document_processing'])
            st.metric("Document Processing Events", doc_processing_count)
    
    # Events over time
    st.header("Events Over Time")
    
    if 'parsed_timestamp' in df.columns:
        # Resample by hour
        df['hour'] = df['parsed_timestamp'].dt.floor('H')
        events_by_hour = df.groupby('hour').size().reset_index(name='count')
        
        # Create time series chart
        fig = px.line(
            events_by_hour, 
            x='hour', 
            y='count',
            title='Events by Hour',
            labels={'hour': 'Time', 'count': 'Number of Events'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Event types breakdown
    if 'event_type' in df.columns:
        st.header("Event Types")
        
        event_counts = df['event_type'].value_counts().reset_index()
        event_counts.columns = ['Event Type', 'Count']
        
        fig = px.pie(
            event_counts, 
            values='Count', 
            names='Event Type',
            title='Event Types Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model usage statistics
    if 'event_type' in df.columns and 'model_usage' in df['event_type'].values:
        st.header("Model Usage")
        
        model_df = df[df['event_type'] == 'model_usage']
        
        if not model_df.empty and 'model_name' in model_df.columns:
            # Model usage by model
            model_counts = model_df['model_name'].value_counts().reset_index()
            model_counts.columns = ['Model', 'Count']
            
            fig = px.bar(
                model_counts, 
                x='Model', 
                y='Count',
                title='Model Usage by Model',
                labels={'Model': 'Model Name', 'Count': 'Number of Calls'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Response time statistics
            if 'duration_ms' in model_df.columns:
                st.subheader("Response Time Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    avg_duration = model_df['duration_ms'].mean()
                    st.metric("Average Response Time (ms)", f"{avg_duration:.2f}")
                
                with col2:
                    max_duration = model_df['duration_ms'].max()
                    st.metric("Max Response Time (ms)", f"{max_duration:.2f}")
                
                # Response time by model
                model_duration = model_df.groupby('model_name')['duration_ms'].mean().reset_index()
                model_duration.columns = ['Model', 'Average Duration (ms)']
                
                fig = px.bar(
                    model_duration, 
                    x='Model', 
                    y='Average Duration (ms)',
                    title='Average Response Time by Model',
                    labels={'Model': 'Model Name', 'Average Duration (ms)': 'Average Duration (ms)'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Document processing statistics
    if 'event_type' in df.columns and 'document_processing' in df['event_type'].values:
        st.header("Document Processing")
        
        doc_df = df[df['event_type'] == 'document_processing']
        
        if not doc_df.empty:
            # Document count statistics
            if 'doc_count' in doc_df.columns:
                total_docs = doc_df['doc_count'].sum()
                st.metric("Total Documents Processed", total_docs)
            
            # Document processing time
            if 'duration_ms' in doc_df.columns:
                avg_duration = doc_df['duration_ms'].mean()
                st.metric("Average Processing Time (ms)", f"{avg_duration:.2f}")
    
    # Thread activity
    if 'event_type' in df.columns and 'thread_activity' in df['event_type'].values:
        st.header("Thread Activity")
        
        thread_df = df[df['event_type'] == 'thread_activity']
        
        if not thread_df.empty and 'action' in thread_df.columns:
            # Thread actions breakdown
            action_counts = thread_df['action'].value_counts().reset_index()
            action_counts.columns = ['Action', 'Count']
            
            fig = px.pie(
                action_counts, 
                values='Count', 
                names='Action',
                title='Thread Actions Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Raw logs table
    st.header("Raw Logs")
    
    # Select columns to display
    display_columns = ['timestamp', 'level', 'component', 'event_type', 'message']
    display_df = df[display_columns] if all(col in df.columns for col in display_columns) else df
    
    # Display the table
    st.dataframe(display_df, use_container_width=True)

if __name__ == "__main__":
    create_dashboard() 