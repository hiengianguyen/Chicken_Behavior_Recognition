import numpy as np

from .grid import OccupancyGrid


class OccupancyExtractor:

    def __init__(self, frame_width=1920, frame_height=1080, rows=8, cols=8):

        self.grid_builder = OccupancyGrid(frame_width, frame_height, rows, cols)

    def extract(self, frame_df):

        centers = []

        for _, row in frame_df.iterrows():

            centers.append((row["x"], row["y"]))

        grid = self.grid_builder.build(centers)

        statistics = self._extract_statistics(grid)

        return {"grid": grid, "statistics": statistics}

    def _extract_statistics(self, grid):

        occupied = grid[grid > 0]

        occupied_cells = int(len(occupied))

        total_cells = grid.size

        occupancy_ratio = occupied_cells / total_cells

        if occupied_cells == 0:

            mean_cell = 0
            std_cell = 0
            max_cell = 0

        else:

            mean_cell = float(np.mean(occupied))

            std_cell = float(np.std(occupied))

            max_cell = int(np.max(occupied))

        return {
            "occupied_cells": occupied_cells,
            "occupancy_ratio": occupancy_ratio,
            "mean_cell_count": mean_cell,
            "std_cell_count": std_cell,
            "max_cell_count": max_cell,
        }
