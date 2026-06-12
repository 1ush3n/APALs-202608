import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def plot_gantt(assigned_tasks, output_path="schedule_gantt.png"):
    """
    Draw a Gantt chart of the schedule.
    assigned_tasks: List of (task_id, station_id, team, start_time, end_time)
    output_path: Path to save the image.
    """
    # 1. Setup Canvas
    # Reduced DPI for memory efficiency
    fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
    
    # 2. Extract Data
    # Y-axis: Station IDs (1..5)
    # X-axis: Time
    
    # Color palette (Simple alternating or random)
    colors = plt.cm.tab20.colors # 20 colors
    
    # 3. Plot Bars
    for i, (task_id, station_id, team, start, end) in enumerate(assigned_tasks):
        duration = end - start
        if duration <= 0: continue
        
        # Station ID is 0-indexed in code, map to 1-indexed for display
        y_pos = station_id + 1 
        
        # Color based on something (e.g., task_id hash or just random)
        color = colors[int(task_id) % len(colors)]
        
        # Barh: (y, width, left, height)
        # edgecolor='none' to remove borders for high density
        ax.barh(y_pos, duration, left=start, height=0.6, color=color, edgecolor='none', align='center')
        
    # 4. Formatting
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Station')
    ax.set_yticks(range(1, 6)) # Assuming 5 stations
    ax.set_yticklabels([f'Station {i}' for i in range(1, 6)])
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.set_title('Assembly Line Schedule (Gantt Chart)')
    
    # 5. Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Gantt chart saved to {output_path}")
