import json

from .camera import Camera
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os


def export_camera_graph(
    cam_list: list[Camera], output_file: str = "results/visuals/camera_graph.json"
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    graph_data = {"nodes": [], "edges": []}

    for cam in cam_list:
        cam_id = cam.camera_idx
        # Add the node (camera) and its real-world anchor
        graph_data["nodes"].append(
            {"id": cam_id, "lon": cam.centroid.x, "lat": cam.centroid.y}
        )

        # Add Overlap Edges
        for ov_cam in cam.overlapping_cameras:
            graph_data["edges"].append(
                {"source": cam_id, "target": ov_cam.camera_idx, "type": "overlap"}
            )

        # Add Adjacency Edges
        for adj_cam in cam.adjacent_cameras:
            graph_data["edges"].append(
                {"source": cam_id, "target": adj_cam.camera_idx, "type": "adjacency"}
            )

    with open(output_file, "w") as f:
        json.dump(graph_data, f, indent=4)
    print(f"Camera graph exported to {output_file}")


def visualize_camera_graph(
    json_file: str, output_file: str = "results/visuals/camera_graph.png"
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # 1. Load the exported data
    with open(json_file, "r") as f:
        data = json.load(f)

    # 2. Initialize an undirected graph
    G = nx.Graph()

    # 3. Add nodes (ignoring lon/lat completely)
    for node in data["nodes"]:
        G.add_node(node["id"])

    # 4. Add edges with their specific types
    for edge in data["edges"]:
        G.add_edge(edge["source"], edge["target"], edge_type=edge["type"])

    # 5. Generate abstract topological positions
    # The spring layout naturally untangles the graph for visual clarity
    positions = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # 6. Set up a clean matplotlib canvas
    plt.figure(figsize=(10, 8))

    # Separate the edges so we can style them differently
    overlap_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "overlap"
    ]
    adjacency_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "adjacency"
    ]

    # Draw the camera nodes
    nx.draw_networkx_nodes(
        G,
        positions,
        node_size=1000,
        node_color="lightblue",
        edgecolors="black",
        linewidths=1.5,
    )

    # Draw Overlap edges (Solid Green)
    nx.draw_networkx_edges(
        G, positions, edgelist=overlap_edges, width=3.0, edge_color="#2ca02c"
    )

    # Draw Adjacency edges (Dashed Orange)
    nx.draw_networkx_edges(
        G,
        positions,
        edgelist=adjacency_edges,
        width=2.5,
        edge_color="#ff7f0e",
        style="dashed",
    )

    # Add the camera IDs inside the nodes
    nx.draw_networkx_labels(G, positions, font_size=12, font_weight="bold")

    # Formatting and styling
    plt.title("Multi-Camera Relational Graph", fontsize=16, fontweight="bold", pad=20)
    plt.axis("off")  # Hide the axes since it's an abstract layout

    # Create a custom legend
    overlap_line = mlines.Line2D(
        [], [], color="#2ca02c", linewidth=3, label="Overlap (Simultaneous)"
    )
    adj_line = mlines.Line2D(
        [],
        [],
        color="#ff7f0e",
        linewidth=2.5,
        linestyle="dashed",
        label="Adjacency (Temporal)",
    )

    # Place legend slightly outside the graph to prevent overlap
    plt.legend(
        handles=[overlap_line, adj_line],
        loc="upper left",
        fontsize=11,
        bbox_to_anchor=(1, 1),
    )

    # Save and show
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Graph saved as '{output_file}'")


def visualize_geospatial_graph(json_file: str, output_file: str = "results/visuals/camera_graph.png"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # 1. Load the exported data
    with open(json_file, "r") as f:
        data = json.load(f)

    # 2. Initialize graph and extract geographic positions
    G = nx.Graph()
    positions = {}
    for node in data["nodes"]:
        cam_id = node["id"]
        G.add_node(cam_id)
        # Use lon as X and lat as Y
        positions[cam_id] = (node["lon"], node["lat"])

    # 3. Add edges with their specific types
    for edge in data["edges"]:
        G.add_edge(edge["source"], edge["target"], edge_type=edge["type"])

    # 4. Set up the canvas
    fig, ax = plt.subplots(figsize=(12, 8))

    overlap_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "overlap"
    ]
    adjacency_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "adjacency"
    ]

    # Draw nodes
    nx.draw_networkx_nodes(
        G, positions, node_size=800, node_color="lightblue", edgecolors="black", ax=ax
    )

    # Draw edges
    nx.draw_networkx_edges(
        G, positions, edgelist=overlap_edges, width=2.5, edge_color="#2ca02c", ax=ax
    )
    nx.draw_networkx_edges(
        G,
        positions,
        edgelist=adjacency_edges,
        width=2.0,
        edge_color="#ff7f0e",
        style="dashed",
        ax=ax,
    )

    # Draw labels
    nx.draw_networkx_labels(G, positions, font_size=11, font_weight="bold", ax=ax)

    # ---------------------------------------------------------
    # 🛠️ THE AXES OVERRIDE
    # NetworkX turns axes off by default. We force them back on here.
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    # Format the tick labels so they don't get squished into scientific notation
    ax.ticklabel_format(useOffset=False, style="plain")

    plt.title(
        "Camera Topology by Real-World Coordinates",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Longitude (X)", fontsize=12, fontweight="bold")
    plt.ylabel("Latitude (Y)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.7)
    # ---------------------------------------------------------

    # Create custom legend
    overlap_line = mlines.Line2D(
        [], [], color="#2ca02c", linewidth=2.5, label="Overlap (Simultaneous)"
    )
    adj_line = mlines.Line2D(
        [],
        [],
        color="#ff7f0e",
        linewidth=2,
        linestyle="dashed",
        label="Adjacency (Temporal)",
    )
    plt.legend(handles=[overlap_line, adj_line], loc="upper right", fontsize=11)

    # Save and show
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Geospatial graph saved as '{output_file}'")
