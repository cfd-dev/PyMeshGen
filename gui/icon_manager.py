"""
PyMeshGen Icon Manager
Provides optimized icons for all GUI functions with consistent styling
"""

import os
import numpy as np
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QBrush, QPolygon
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QStyle, QApplication


class IconManager:
    """Icon manager for PyMeshGen - provides optimized, consistent icons"""
    
    def __init__(self, project_root):
        self.project_root = project_root
        self.icon_cache = {}
        self.color_scheme = {
            'primary': '#0078D4',      # Microsoft blue
            'secondary': '#107C10',    # Microsoft green
            'accent': '#D83B01',       # Microsoft orange
            'warning': '#FFD34E',      # Warning yellow
            'danger': '#E81123',       # Danger red
            'success': '#107C10',      # Success green
            'info': '#00BCF2',         # Info blue
            'dark': '#201F1E',         # Dark gray
            'light': '#FFFFFF',        # White
            'gray': '#CCCCCC'          # Gray
        }
    
    def get_icon(self, icon_name):
        """Get icon by name, either from cache or create it"""
        if icon_name in self.icon_cache:
            return self.icon_cache[icon_name]
        
        # Try to get from standard icon system first
        icon = self._get_standard_icon(icon_name)
        if icon and not icon.isNull():
            self.icon_cache[icon_name] = icon
            return icon
        
        # Create custom icon if standard icon not available
        icon = self._create_custom_icon(icon_name)
        self.icon_cache[icon_name] = icon
        return icon
    
    def _get_standard_icon(self, icon_name):
        """Get standard Qt icon"""
        icon_map = {
            # File operations
            'document-new': QStyle.SP_FileIcon,
            'document-open': QStyle.SP_DirOpenIcon,
            'document-save': QStyle.SP_DialogSaveButton,
            'document-import': QStyle.SP_ArrowDown,
            'document-export': QStyle.SP_ArrowUp,
            'import': QStyle.SP_ArrowDown,
            'export': QStyle.SP_ArrowUp,
            
            # View operations
            'zoom-fit-best': QStyle.SP_ComputerIcon,
            'zoom-in': None,  # 使用自定义图标
            'zoom-out': None,  # 使用自定义图标
            'view-refresh': QStyle.SP_BrowserReload,
            'view-fullscreen': QStyle.SP_TitleBarMaxButton,
            
            # Edit operations
            'edit-delete': QStyle.SP_TrashIcon,
            'edit-clear': QStyle.SP_LineEditClearButton,
            'system-run': QStyle.SP_MediaPlay,
            'run': QStyle.SP_MediaPlay,
            
            # Configuration and settings
            'configure': QStyle.SP_FileDialogDetailedView,
            'document-properties': QStyle.SP_FileDialogDetailedView,
            
            # Help operations
            'help-contents': QStyle.SP_MessageBoxInformation,
            'help-about': QStyle.SP_MessageBoxQuestion,
            'help-faq': QStyle.SP_MessageBoxQuestion,
            'help-keyboard-shortcuts': QStyle.SP_DialogHelpButton,
        }
        
        if icon_name in icon_map:
            app = QApplication.instance()
            if app:
                icon_value = icon_map[icon_name]
                if icon_value is not None:
                    return app.style().standardIcon(icon_value)
        
        return QIcon()  # Return empty icon if not found
    
    def _create_custom_icon(self, icon_name):
        """Create custom icon for specific functions"""
        # Define icon dimensions
        size = 32
        
        # Create pixmap
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        # Create painter
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get colors
        primary_color = QColor(self.color_scheme['primary'])
        secondary_color = QColor(self.color_scheme['secondary'])
        accent_color = QColor(self.color_scheme['accent'])

        # Draw different icons based on icon_name
        if icon_name in ['mesh-generate', 'generate-mesh', 'mesh_generate', 'mesh-generate']:
            self._draw_mesh_generate_icon(painter, size, primary_color)
        elif icon_name in ['mesh-quality', 'check-quality', 'mesh_quality', 'quality']:
            self._draw_mesh_quality_icon(painter, size, secondary_color)
        elif icon_name in ['mesh-smooth', 'smooth-mesh', 'mesh_smooth', 'smooth']:
            self._draw_mesh_smooth_icon(painter, size, primary_color)
        elif icon_name in ['mesh-optimize', 'optimize-mesh', 'mesh_optimize', 'optimize']:
            self._draw_mesh_optimize_icon(painter, size, accent_color)
        elif icon_name in ['mesh-dimension', 'mesh_dimension', 'mesh-dim', 'dimension']:
            self._draw_mesh_dimension_icon(painter, size, primary_color)
        elif icon_name in ['part-params', 'edit-part', 'part_params', 'part-params']:
            self._draw_part_params_icon(painter, size, primary_color)
        elif icon_name in ['config-import', 'import-config', 'config_import', 'import']:
            self._draw_config_import_icon(painter, size, secondary_color)
        elif icon_name in ['config-export', 'export-config', 'config_export', 'export']:
            self._draw_config_export_icon(painter, size, secondary_color)
        elif icon_name in ['boundary-condition', 'boundary']:
            self._draw_boundary_condition_icon(painter, size, accent_color)
        elif icon_name in ['extract-boundary', 'extract_boundary']:
            self._draw_extract_boundary_icon(painter, size, primary_color)
        elif icon_name in ['statistics', 'stats']:
            self._draw_statistics_icon(painter, size, primary_color)
        elif icon_name == 'report':
            self._draw_report_icon(painter, size, secondary_color)
        elif icon_name == 'view-x-pos':
            self._draw_view_x_pos_icon(painter, size, primary_color)
        elif icon_name == 'view-x-neg':
            self._draw_view_x_neg_icon(painter, size, primary_color)
        elif icon_name == 'view-y-pos':
            self._draw_view_y_pos_icon(painter, size, secondary_color)
        elif icon_name == 'view-y-neg':
            self._draw_view_y_neg_icon(painter, size, secondary_color)
        elif icon_name == 'view-z-pos':
            self._draw_view_z_pos_icon(painter, size, accent_color)
        elif icon_name == 'view-z-neg':
            self._draw_view_z_neg_icon(painter, size, accent_color)
        elif icon_name == 'view-iso':
            self._draw_view_iso_icon(painter, size, primary_color)
        elif icon_name == 'surface':
            self._draw_surface_icon(painter, size, primary_color)
        elif icon_name == 'wireframe':
            self._draw_wireframe_icon(painter, size, secondary_color)
        elif icon_name == 'surface-wireframe':
            self._draw_surface_wireframe_icon(painter, size, accent_color)
        elif icon_name in ['geom-create', 'geometry-create']:
            self._draw_geometry_create_icon(painter, size, primary_color)
        elif icon_name in ['geom-point', 'geometry-point']:
            self._draw_geometry_point_icon(painter, size, primary_color)
        elif icon_name in ['geom-line', 'geometry-line']:
            self._draw_geometry_line_icon(painter, size, primary_color)
        elif icon_name in ['geom-circle', 'geometry-circle', 'geom-arc']:
            self._draw_geometry_circle_icon(painter, size, primary_color)
        elif icon_name in ['geom-curve', 'geometry-curve']:
            self._draw_geometry_curve_icon(painter, size, primary_color)
        elif icon_name in ['geom-polyline', 'geometry-polyline']:
            self._draw_geometry_polyline_icon(painter, size, primary_color)
        elif icon_name in ['geom-rectangle', 'geometry-rectangle']:
            self._draw_geometry_rectangle_icon(painter, size, primary_color)
        elif icon_name in ['geom-polygon', 'geometry-polygon']:
            self._draw_geometry_polygon_icon(painter, size, primary_color)
        elif icon_name in ['geom-ellipse', 'geometry-ellipse']:
            self._draw_geometry_ellipse_icon(painter, size, primary_color)
        elif icon_name in ['geom-box', 'geometry-box']:
            self._draw_geometry_box_icon(painter, size, accent_color)
        elif icon_name in ['geom-sphere', 'geometry-sphere']:
            self._draw_geometry_sphere_icon(painter, size, accent_color)
        elif icon_name in ['geom-cylinder', 'geometry-cylinder']:
            self._draw_geometry_cylinder_icon(painter, size, accent_color)
        elif icon_name in ['line-mesh-generate', 'line_mesh_generate', 'line-discretize']:
            self._draw_line_mesh_icon(painter, size, primary_color)
        elif icon_name in ['create-region', 'create_region']:
            self._draw_create_region_icon(painter, size, secondary_color)
        elif icon_name == 'zoom-in':
            self._draw_zoom_in_icon(painter, size, primary_color)
        elif icon_name == 'zoom-out':
            self._draw_zoom_out_icon(painter, size, primary_color)
        else:
            # Default fallback: draw a generic icon
            self._draw_generic_icon(painter, size, primary_color)
        
        painter.end()
        
        return QIcon(pixmap)
    
    def _draw_mesh_generate_icon(self, painter, size, color):
        """Draw mesh generation icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Draw a simple grid pattern (4x4)
        cell_size = size // 4
        for i in range(5):
            # Horizontal lines
            painter.drawLine(0, i * cell_size, size, i * cell_size)
            # Vertical lines
            painter.drawLine(i * cell_size, 0, i * cell_size, size)
        
        # Draw small triangles in some cells to represent mesh elements
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        
        # Draw a few triangles in the grid
        triangle_size = cell_size - 2
        # Top-left triangle
        painter.drawPolygon(QPolygon([
            QPoint(2, 2),
            QPoint(2 + triangle_size, 2),
            QPoint(2, 2 + triangle_size)
        ]))
        # Bottom-right triangle
        painter.drawPolygon(QPolygon([
            QPoint(size - 2, size - 2),
            QPoint(size - 2 - triangle_size, size - 2),
            QPoint(size - 2, size - 2 - triangle_size)
        ]))
    
    def _draw_mesh_quality_icon(self, painter, size, color):
        """Draw mesh quality icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)

        # Draw a diamond shape representing quality
        center_x, center_y = size // 2, size // 2
        radius = size // 3

        points = [
            (center_x, center_y - radius),  # Top
            (center_x + radius, center_y),  # Right
            (center_x, center_y + radius),  # Bottom
            (center_x - radius, center_y)   # Left
        ]

        # Convert list of tuples to QPolygon
        qpoints = [QPoint(int(x), int(y)) for x, y in points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)

        # Draw checkmark inside
        painter.setBrush(color)
        painter.drawLine(size//2 - 4, size//2, size//2, size//2 + 4)
        painter.drawLine(size//2, size//2 + 4, size//2 + 6, size//2 - 2)
    
    def _draw_mesh_smooth_icon(self, painter, size, color):
        """Draw mesh smoothing icon - showing jagged to smooth transition"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Draw jagged line on left (before smoothing)
        pen.setColor(QColor('#CCCCCC'))
        painter.setPen(pen)
        jagged_points = [
            (4, size // 2),
            (8, size // 2 - 6),
            (12, size // 2 + 4),
            (16, size // 2 - 3),
            (20, size // 2 + 5),
            (24, size // 2 - 2)
        ]
        for i in range(len(jagged_points) - 1):
            painter.drawLine(jagged_points[i][0], jagged_points[i][1], 
                           jagged_points[i+1][0], jagged_points[i+1][1])
        
        # Draw arrow pointing right
        pen.setColor(color)
        painter.setPen(pen)
        painter.setBrush(color)
        arrow_x = size // 2
        arrow_y = size // 2
        painter.drawLine(arrow_x - 4, arrow_y, arrow_x + 4, arrow_y)
        painter.drawPolygon(QPolygon([
            QPoint(arrow_x + 4, arrow_y),
            QPoint(arrow_x, arrow_y - 3),
            QPoint(arrow_x, arrow_y + 3)
        ]))
        
        # Draw smooth curve on right (after smoothing)
        pen.setColor(color)
        painter.setPen(pen)
        smooth_points = []
        start_x = arrow_x + 8
        for i in range(8):
            x = start_x + i * 3
            y = size // 2 + int(4 * np.sin(i * 0.5))
            smooth_points.append((x, y))
        
        for i in range(len(smooth_points) - 1):
            painter.drawLine(smooth_points[i][0], smooth_points[i][1], 
                           smooth_points[i+1][0], smooth_points[i+1][1])
    
    def _draw_mesh_optimize_icon(self, painter, size, color):
        """Draw mesh optimization icon - gear with upward arrow"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        
        center_x, center_y = size // 2, size // 2
        radius = size // 3
        
        # Draw outer gear circle
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # Draw gear teeth
        painter.setBrush(color)
        for i in range(8):
            painter.save()
            painter.translate(center_x, center_y)
            painter.rotate(i * 45)
            painter.drawRect(-2, -radius + 2, 4, 5)
            painter.restore()
        
        # Draw inner circle
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center_x - radius // 3, center_y - radius // 3, 
                          radius * 2 // 3, radius * 2 // 3)
        
        # Draw upward arrow in center
        painter.setPen(QPen(color, 2))
        painter.setBrush(color)
        arrow_top = center_y - radius // 3 + 2
        arrow_bottom = center_y + radius // 3 - 2
        
        # Arrow shaft
        painter.drawLine(center_x, arrow_bottom, center_x, arrow_top + 3)
        
        # Arrow head
        painter.drawPolygon(QPolygon([
            QPoint(center_x, arrow_top),
            QPoint(center_x - 4, arrow_top + 5),
            QPoint(center_x + 4, arrow_top + 5)
        ]))

    def _draw_mesh_dimension_icon(self, painter, size, color):
        """Draw mesh dimension icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        offset = 4
        rect_size = size - 12
        back_x = 6
        back_y = 6
        front_x = back_x + offset
        front_y = back_y + offset

        painter.drawRect(back_x, back_y, rect_size, rect_size)
        painter.drawRect(front_x, front_y, rect_size, rect_size)

        painter.drawLine(back_x, back_y, front_x, front_y)
        painter.drawLine(back_x + rect_size, back_y, front_x + rect_size, front_y)
        painter.drawLine(back_x, back_y + rect_size, front_x, front_y + rect_size)

        font = painter.font()
        font.setPointSize(6)
        painter.setFont(font)
        painter.drawText(2, size - 4, "2D/3D")
    
    def _draw_part_params_icon(self, painter, size, color):
        """Draw part parameters icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Draw gear icon
        center_x, center_y = size // 2, size // 2
        radius = size // 4
        
        # Draw outer circle
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # Draw inner gear teeth
        painter.setBrush(color)
        for i in range(6):
            painter.save()
            painter.translate(center_x, center_y)
            painter.rotate(i * 60)
            painter.drawRect(-2, -radius + 2, 4, 4)  # Small rectangle as gear tooth
            painter.restore()
    
    def _draw_config_import_icon(self, painter, size, color):
        """Draw config import icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Draw folder
        painter.drawRect(5, size//2, size - 10, size//2 - 5)
        painter.drawRect(10, size//2 - 10, size - 20, 10)
        
        # Draw import arrow (downward arrow)
        painter.drawLine(size - 10, 10, size - 10, size//2 - 5)
        painter.drawLine(size - 10, size//2 - 5, size - 15, size//2 - 10)
        painter.drawLine(size - 10, size//2 - 5, size - 5, size//2 - 10)
    
    def _draw_config_export_icon(self, painter, size, color):
        """Draw config export icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Draw folder
        painter.drawRect(5, size//2, size - 10, size//2 - 5)
        painter.drawRect(10, size//2 - 10, size - 20, 10)
        
        # Draw export arrow (arrow pointing outward from folder to the right)
        folder_center_y = size//2 + (size//2 - 5)//2
        painter.drawLine(size - 20, folder_center_y, size - 6, folder_center_y)
        painter.drawLine(size - 6, folder_center_y, size - 10, folder_center_y - 4)
        painter.drawLine(size - 6, folder_center_y, size - 10, folder_center_y + 4)
    
    def _draw_boundary_condition_icon(self, painter, size, color):
        """Draw boundary condition icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)

        # Draw boundary rectangle
        painter.drawRect(5, 5, size - 10, size - 10)

        # Draw boundary markers
        marker_positions = [
            (size//4, 5), (size*3//4, 5),  # Top
            (5, size//4), (size - 5, size//4),  # Left/right
            (size//4, size - 5), (size*3//4, size - 5),  # Bottom
            (5, size*3//4), (size - 5, size*3//4)   # Left/right
        ]

        painter.setBrush(color)
        for x, y in marker_positions:
            painter.drawRect(int(x) - 2, int(y) - 2, 4, 4)
    
    def _draw_extract_boundary_icon(self, painter, size, color):
        """Draw extract boundary icon"""
        # Draw a simple mesh (two triangles)
        pen = QPen(QColor('#CCCCCC'), 1)
        painter.setPen(pen)
        
        # Define mesh points
        p1 = (size//4, size//4)
        p2 = (size*3//4, size//4)
        p3 = (size//2, size*3//4)
        
        # Draw mesh triangles
        painter.drawLine(p1[0], p1[1], p2[0], p2[1])
        painter.drawLine(p2[0], p2[1], p3[0], p3[1])
        painter.drawLine(p3[0], p3[1], p1[0], p1[1])
        
        # Draw boundary edges with highlight color
        boundary_pen = QPen(color, 3)
        painter.setPen(boundary_pen)
        
        # Draw outer boundary (highlighted)
        painter.drawLine(p1[0], p1[1], p2[0], p2[1])
        painter.drawLine(p2[0], p2[1], p3[0], p3[1])
        painter.drawLine(p3[0], p3[1], p1[0], p1[1])
        
        # Draw extraction arrow pointing outward
        arrow_pen = QPen(color, 2)
        painter.setPen(arrow_pen)
        painter.setBrush(color)
        
        # Draw arrow from center to right
        center_x, center_y = size//2, size//2
        arrow_start_x = center_x + 5
        arrow_end_x = size - 8
        
        # Arrow shaft
        painter.drawLine(arrow_start_x, center_y, arrow_end_x, center_y)
        
        # Arrow head
        arrow_points = [
            (arrow_end_x, center_y),
            (arrow_end_x - 4, center_y - 4),
            (arrow_end_x - 4, center_y + 4)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in arrow_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
    
    def _draw_statistics_icon(self, painter, size, color):
        """Draw statistics icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Draw chart bars
        bar_width = size // 8
        bar_heights = [size//4, size*2//4, size*3//4, size//2, size*3//4]
        spacing = size // 8
        
        for i, height in enumerate(bar_heights):
            x = 5 + i * (bar_width + spacing)
            y = size - 5 - height
            painter.drawRect(x, y, bar_width, height)
    
    def _draw_report_icon(self, painter, size, color):
        """Draw report icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Draw document
        painter.drawRect(5, 5, size - 10, size - 10)
        
        # Draw lines representing text
        for i in range(3):
            y_pos = 10 + i * 6
            painter.drawLine(10, y_pos, size - 10, y_pos)
        
        # Draw chart icon overlay
        bar_width = 3
        for i in range(3):
            x = size - 15 + i * 4
            height = 8 - i * 2
            painter.drawRect(x, size - 15, bar_width, height)
    
    def _draw_view_x_pos_icon(self, painter, size, color):
        """Draw X+ view icon - simplified arrow pointing right"""
        pen = QPen(color, 3)
        painter.setPen(pen)
        painter.setBrush(color)
        
        center_x = size // 2
        center_y = size // 2
        
        # Draw horizontal arrow pointing right
        # Arrow shaft
        painter.drawLine(6, center_y, size - 6, center_y)
        
        # Arrow head
        arrow_points = [
            (size - 6, center_y),
            (size - 10, center_y - 4),
            (size - 10, center_y + 4)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in arrow_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw X label
        painter.setPen(color)
        painter.setFont(painter.font())
        painter.drawText(center_x - 4, center_y - 8, "X+")
    
    def _draw_view_x_neg_icon(self, painter, size, color):
        """Draw X- view icon - simplified arrow pointing left"""
        pen = QPen(color, 3)
        painter.setPen(pen)
        painter.setBrush(color)
        
        center_x = size // 2
        center_y = size // 2
        
        # Draw horizontal arrow pointing left
        # Arrow shaft
        painter.drawLine(size - 6, center_y, 6, center_y)
        
        # Arrow head
        arrow_points = [
            (6, center_y),
            (10, center_y - 4),
            (10, center_y + 4)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in arrow_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw X label
        painter.setPen(color)
        painter.drawText(center_x - 4, center_y - 8, "X-")
    
    def _draw_view_y_pos_icon(self, painter, size, color):
        """Draw Y+ view icon - simplified arrow pointing up"""
        pen = QPen(color, 3)
        painter.setPen(pen)
        painter.setBrush(color)
        
        center_x = size // 2
        center_y = size // 2
        
        # Draw vertical arrow pointing up
        # Arrow shaft
        painter.drawLine(center_x, size - 6, center_x, 6)
        
        # Arrow head
        arrow_points = [
            (center_x, 6),
            (center_x - 4, 10),
            (center_x + 4, 10)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in arrow_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw Y label
        painter.setPen(color)
        painter.drawText(center_x + 6, center_y + 4, "Y+")
    
    def _draw_view_y_neg_icon(self, painter, size, color):
        """Draw Y- view icon - simplified arrow pointing down"""
        pen = QPen(color, 3)
        painter.setPen(pen)
        painter.setBrush(color)
        
        center_x = size // 2
        center_y = size // 2
        
        # Draw vertical arrow pointing down
        # Arrow shaft
        painter.drawLine(center_x, 6, center_x, size - 6)
        
        # Arrow head
        arrow_points = [
            (center_x, size - 6),
            (center_x - 4, size - 10),
            (center_x + 4, size - 10)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in arrow_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw Y label
        painter.setPen(color)
        painter.drawText(center_x + 6, center_y + 4, "Y-")
    
    def _draw_view_z_pos_icon(self, painter, size, color):
        """Draw Z+ view icon - simplified circle with dot (pointing out)"""
        pen = QPen(color, 3)
        painter.setPen(pen)
        painter.setBrush(color)
        
        center_x = size // 2
        center_y = size // 2
        
        # Draw circle with dot for Z+ (pointing out)
        painter.drawEllipse(center_x - 8, center_y - 8, 16, 16)
        painter.drawEllipse(center_x - 3, center_y - 3, 6, 6)
        
        # Draw Z label
        painter.setPen(color)
        painter.drawText(center_x + 10, center_y + 4, "Z+")
    
    def _draw_view_z_neg_icon(self, painter, size, color):
        """Draw Z- view icon - simplified circle with X (pointing in)"""
        pen = QPen(color, 3)
        painter.setPen(pen)
        painter.setBrush(color)
        
        center_x = size // 2
        center_y = size // 2
        
        # Draw circle with X for Z- (pointing in)
        painter.drawEllipse(center_x - 8, center_y - 8, 16, 16)
        painter.drawLine(center_x - 5, center_y - 5, center_x + 5, center_y + 5)
        painter.drawLine(center_x + 5, center_y - 5, center_x - 5, center_y + 5)
        
        # Draw Z label
        painter.setPen(color)
        painter.drawText(center_x + 10, center_y + 4, "Z-")
    
    def _draw_view_iso_icon(self, painter, size, color):
        """Draw isometric view icon - simplified 3D cube"""
        pen = QPen(color, 3)
        painter.setPen(pen)
        painter.setBrush(color)
        
        center_x = size // 2
        center_y = size // 2
        cube_size = size // 3
        
        # Draw isometric cube with filled faces
        # Draw top face (lighter)
        top_color = QColor(color)
        top_color.setAlpha(180)
        painter.setBrush(top_color)
        top_points = [
            (center_x, center_y - cube_size),
            (center_x + cube_size * 0.866, center_y - cube_size * 0.5),
            (center_x, center_y),
            (center_x - cube_size * 0.866, center_y - cube_size * 0.5)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in top_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw left face (darker)
        left_color = QColor(color)
        left_color.setAlpha(120)
        painter.setBrush(left_color)
        left_points = [
            (center_x - cube_size * 0.866, center_y - cube_size * 0.5),
            (center_x, center_y),
            (center_x, center_y + cube_size),
            (center_x - cube_size * 0.866, center_y + cube_size * 0.5)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in left_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw right face (medium)
        right_color = QColor(color)
        right_color.setAlpha(150)
        painter.setBrush(right_color)
        right_points = [
            (center_x + cube_size * 0.866, center_y - cube_size * 0.5),
            (center_x, center_y),
            (center_x, center_y + cube_size),
            (center_x + cube_size * 0.866, center_y + cube_size * 0.5)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in right_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw ISO label
        painter.setPen(color)
        painter.drawText(center_x - 8, size - 2, "ISO")
    
    def _draw_surface_icon(self, painter, size, color):
        """Draw surface mode icon - filled 3D cube"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        center_x = size // 2
        center_y = size // 2
        cube_size = size // 3
        
        # Draw filled isometric cube
        # Draw top face (lighter)
        top_color = QColor(color)
        top_color.setAlpha(200)
        painter.setBrush(top_color)
        top_points = [
            (center_x, center_y - cube_size),
            (center_x + cube_size * 0.866, center_y - cube_size * 0.5),
            (center_x, center_y),
            (center_x - cube_size * 0.866, center_y - cube_size * 0.5)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in top_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw left face (darker)
        left_color = QColor(color)
        left_color.setAlpha(140)
        painter.setBrush(left_color)
        left_points = [
            (center_x - cube_size * 0.866, center_y - cube_size * 0.5),
            (center_x, center_y),
            (center_x, center_y + cube_size),
            (center_x - cube_size * 0.866, center_y + cube_size * 0.5)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in left_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw right face (medium)
        right_color = QColor(color)
        right_color.setAlpha(170)
        painter.setBrush(right_color)
        right_points = [
            (center_x + cube_size * 0.866, center_y - cube_size * 0.5),
            (center_x, center_y),
            (center_x, center_y + cube_size),
            (center_x + cube_size * 0.866, center_y + cube_size * 0.5)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in right_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
    
    def _draw_wireframe_icon(self, painter, size, color):
        """Draw wireframe mode icon - outlined 3D cube"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        center_x = size // 2
        center_y = size // 2
        cube_size = size // 3

        # Draw isometric cube outline only
        
        # Draw top face outline
        top_points = [
            (center_x, center_y - cube_size),
            (center_x + cube_size * 0.866, center_y - cube_size * 0.5),
            (center_x, center_y),
            (center_x - cube_size * 0.866, center_y - cube_size * 0.5)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in top_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw left face outline
        left_points = [
            (center_x - cube_size * 0.866, center_y - cube_size * 0.5),
            (center_x, center_y),
            (center_x, center_y + cube_size),
            (center_x - cube_size * 0.866, center_y + cube_size * 0.5)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in left_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw right face outline
        right_points = [
            (center_x + cube_size * 0.866, center_y - cube_size * 0.5),
            (center_x, center_y),
            (center_x, center_y + cube_size),
            (center_x + cube_size * 0.866, center_y + cube_size * 0.5)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in right_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
    
    def _draw_surface_wireframe_icon(self, painter, size, color):
        """Draw surface+wireframe mode icon - filled 3D cube with strong edges"""
        pen = QPen(color, 3)
        painter.setPen(pen)
        
        center_x = size // 2
        center_y = size // 2
        cube_size = size // 3
        
        # Draw filled isometric cube with strong edge lines
        # Draw top face (lighter)
        top_color = QColor(color)
        top_color.setAlpha(180)
        painter.setBrush(top_color)
        top_points = [
            (center_x, center_y - cube_size),
            (center_x + cube_size * 0.866, center_y - cube_size * 0.5),
            (center_x, center_y),
            (center_x - cube_size * 0.866, center_y - cube_size * 0.5)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in top_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw left face (darker)
        left_color = QColor(color)
        left_color.setAlpha(120)
        painter.setBrush(left_color)
        left_points = [
            (center_x - cube_size * 0.866, center_y - cube_size * 0.5),
            (center_x, center_y),
            (center_x, center_y + cube_size),
            (center_x - cube_size * 0.866, center_y + cube_size * 0.5)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in left_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw right face (medium)
        right_color = QColor(color)
        right_color.setAlpha(150)
        painter.setBrush(right_color)
        right_points = [
            (center_x + cube_size * 0.866, center_y - cube_size * 0.5),
            (center_x, center_y),
            (center_x, center_y + cube_size),
            (center_x + cube_size * 0.866, center_y + cube_size * 0.5)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in right_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw strong edge lines on top
        painter.setPen(QPen(color, 2))
        painter.drawLine(int(center_x), int(center_y - cube_size), int(center_x), int(center_y))
        painter.drawLine(int(center_x), int(center_y), int(center_x - cube_size * 0.866), int(center_y + cube_size * 0.5))
        painter.drawLine(int(center_x), int(center_y), int(center_x + cube_size * 0.866), int(center_y + cube_size * 0.5))

    def _draw_geometry_create_icon(self, painter, size, color):
        """Draw create geometry icon - combined point, line, and shape"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        painter.drawLine(6, size - 8, size - 8, 8)
        painter.drawRect(6, 6, size // 3, size // 3)
        painter.setBrush(QBrush(color))
        dot_radius = max(2, size // 8)
        painter.drawEllipse(size - 10 - dot_radius, size - 10 - dot_radius, dot_radius * 2, dot_radius * 2)

    def _draw_geometry_point_icon(self, painter, size, color):
        """Draw point icon"""
        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(color))
        radius = max(2, size // 6)
        painter.drawEllipse(size // 2 - radius, size // 2 - radius, radius * 2, radius * 2)

    def _draw_geometry_line_icon(self, painter, size, color):
        """Draw line icon"""
        painter.setPen(QPen(color, 2))
        painter.drawLine(6, size - 6, size - 6, 6)

    def _draw_line_mesh_icon(self, painter, size, color):
        """Draw line mesh icon - a line with mesh nodes along it"""
        from PyQt5.QtCore import QPoint

        painter.setPen(QPen(color, 2))

        x1, y1 = 6, size - 6
        x2, y2 = size - 6, 6

        painter.drawLine(x1, y1, x2, y2)

        painter.setBrush(color)

        num_points = 5
        for i in range(num_points + 1):
            t = i / num_points
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            painter.drawEllipse(QPoint(x, y), 2, 2)

    def _draw_geometry_circle_icon(self, painter, size, color):
        """Draw circle/arc icon"""
        painter.setPen(QPen(color, 2))
        rect = (6, 6, size - 12, size - 12)
        painter.drawArc(*rect, 30 * 16, 300 * 16)

    def _draw_geometry_curve_icon(self, painter, size, color):
        """Draw curve icon"""
        from PyQt5.QtGui import QPainterPath
        painter.setPen(QPen(color, 2))
        path = QPainterPath()
        path.moveTo(6, size - 8)
        path.cubicTo(size // 3, 6, size * 2 // 3, size - 6, size - 6, 8)
        painter.drawPath(path)

    def _draw_geometry_polyline_icon(self, painter, size, color):
        """Draw polyline icon"""
        painter.setPen(QPen(color, 2))
        points = [
            QPoint(6, size - 8),
            QPoint(size // 3, size // 3),
            QPoint(size * 2 // 3, size - 10),
            QPoint(size - 6, 8)
        ]
        painter.drawPolyline(QPolygon(points))

    def _draw_geometry_rectangle_icon(self, painter, size, color):
        """Draw rectangle icon"""
        painter.setPen(QPen(color, 2))
        painter.drawRect(6, 6, size - 12, size - 12)

    def _draw_geometry_polygon_icon(self, painter, size, color):
        """Draw polygon icon"""
        painter.setPen(QPen(color, 2))
        points = [
            QPoint(size // 2, 6),
            QPoint(size - 6, size // 3),
            QPoint(size * 3 // 4, size - 6),
            QPoint(size // 4, size - 6),
            QPoint(6, size // 3)
        ]
        painter.drawPolygon(QPolygon(points))

    def _draw_geometry_ellipse_icon(self, painter, size, color):
        """Draw ellipse icon"""
        painter.setPen(QPen(color, 2))
        painter.drawEllipse(6, size // 4, size - 12, size // 2)

    def _draw_geometry_box_icon(self, painter, size, color):
        """Draw box icon"""
        painter.setPen(QPen(color, 2))
        center_x = size // 2
        center_y = size // 2
        cube_size = size // 3
        top_points = [
            QPoint(center_x, center_y - cube_size),
            QPoint(int(center_x + cube_size * 0.866), int(center_y - cube_size * 0.5)),
            QPoint(center_x, center_y),
            QPoint(int(center_x - cube_size * 0.866), int(center_y - cube_size * 0.5))
        ]
        painter.drawPolygon(QPolygon(top_points))
        painter.drawLine(top_points[0], QPoint(center_x, center_y + cube_size))
        painter.drawLine(top_points[1], QPoint(int(center_x + cube_size * 0.866), int(center_y + cube_size * 0.5)))
        painter.drawLine(top_points[3], QPoint(int(center_x - cube_size * 0.866), int(center_y + cube_size * 0.5)))
        bottom_points = [
            QPoint(center_x, center_y + cube_size),
            QPoint(int(center_x + cube_size * 0.866), int(center_y + cube_size * 0.5)),
            QPoint(center_x, center_y),
            QPoint(int(center_x - cube_size * 0.866), int(center_y + cube_size * 0.5))
        ]
        painter.drawPolygon(QPolygon(bottom_points))

    def _draw_geometry_sphere_icon(self, painter, size, color):
        """Draw sphere icon"""
        painter.setPen(QPen(color, 2))
        painter.drawEllipse(6, 6, size - 12, size - 12)
        painter.drawArc(6, size // 4, size - 12, size // 2, 0, 180 * 16)

    def _draw_geometry_cylinder_icon(self, painter, size, color):
        """Draw cylinder icon"""
        painter.setPen(QPen(color, 2))
        top_rect = (6, 6, size - 12, size // 3)
        bottom_rect = (6, size - size // 3 - 6, size - 12, size // 3)
        painter.drawEllipse(*top_rect)
        painter.drawArc(*bottom_rect, 0, 180 * 16)
        painter.drawLine(6, 6 + size // 6, 6, size - size // 6 - 6)
        painter.drawLine(size - 6, 6 + size // 6, size - 6, size - size // 6 - 6)
    
    def _draw_create_region_icon(self, painter, size, color):
        """Draw create region icon - polygon with arrows showing direction"""
        painter.setPen(QPen(color, 2))
        
        # Draw a closed polygon (hexagon)
        center_x, center_y = size // 2, size // 2
        radius = size // 3
        
        points = []
        for i in range(6):
            angle = i * 60 - 30  # Start from -30 degrees
            x = center_x + radius * 0.866 * np.cos(np.radians(angle))
            y = center_y + radius * 0.866 * np.sin(np.radians(angle))
            points.append((int(x), int(y)))
        
        # Draw polygon edges
        for i in range(len(points)):
            next_i = (i + 1) % len(points)
            painter.drawLine(points[i][0], points[i][1], points[next_i][0], points[next_i][1])
        
        # Draw direction arrows on edges
        arrow_color = QColor(color)
        painter.setPen(QPen(arrow_color, 2))
        painter.setBrush(arrow_color)
        
        # Draw arrows on every other edge
        for i in range(0, len(points), 2):
            next_i = (i + 1) % len(points)
            
            # Calculate edge midpoint
            mid_x = (points[i][0] + points[next_i][0]) // 2
            mid_y = (points[i][1] + points[next_i][1]) // 2
            
            # Calculate edge direction
            dx = points[next_i][0] - points[i][0]
            dy = points[next_i][1] - points[i][1]
            edge_length = np.sqrt(dx*dx + dy*dy)
            
            if edge_length > 0:
                dx /= edge_length
                dy /= edge_length
                
                # Draw small arrow at midpoint
                arrow_size = 4
                arrow_points = [
                    QPoint(mid_x, mid_y),
                    QPoint(int(mid_x - dx * arrow_size + dy * arrow_size * 0.5), 
                           int(mid_y - dy * arrow_size - dx * arrow_size * 0.5)),
                    QPoint(int(mid_x - dx * arrow_size - dy * arrow_size * 0.5), 
                           int(mid_y - dy * arrow_size + dx * arrow_size * 0.5))
                ]
                painter.drawPolygon(QPolygon(arrow_points))
    
    def _draw_zoom_in_icon(self, painter, size, color):
        """Draw zoom in icon - magnifying glass with plus sign"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        
        # Draw magnifying glass circle
        center_x, center_y = size // 2, size // 2
        radius = size // 3
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # Draw magnifying glass handle
        handle_start_x = center_x + radius * 0.707
        handle_start_y = center_y + radius * 0.707
        handle_end_x = center_x + radius * 1.414
        handle_end_y = center_y + radius * 1.414
        painter.drawLine(int(handle_start_x), int(handle_start_y), int(handle_end_x), int(handle_end_y))
        
        # Draw plus sign inside the glass
        plus_size = radius // 2
        painter.drawLine(center_x - plus_size, center_y, center_x + plus_size, center_y)  # Horizontal
        painter.drawLine(center_x, center_y - plus_size, center_x, center_y + plus_size)  # Vertical
    
    def _draw_zoom_out_icon(self, painter, size, color):
        """Draw zoom out icon - magnifying glass with minus sign"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        
        # Draw magnifying glass circle
        center_x, center_y = size // 2, size // 2
        radius = size // 3
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # Draw magnifying glass handle
        handle_start_x = center_x + radius * 0.707
        handle_start_y = center_y + radius * 0.707
        handle_end_x = center_x + radius * 1.414
        handle_end_y = center_y + radius * 1.414
        painter.drawLine(int(handle_start_x), int(handle_start_y), int(handle_end_x), int(handle_end_y))
        
        # Draw minus sign inside the glass
        minus_size = radius // 2
        painter.drawLine(center_x - minus_size, center_y, center_x + minus_size, center_y)  # Horizontal
    
    def _draw_generic_icon(self, painter, size, color):
        """Draw a generic icon as fallback"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Draw a simple circle
        painter.drawEllipse(5, 5, size - 10, size - 10)
        
        # Draw a simple geometric shape
        painter.drawLine(10, size//2, size - 10, size//2)  # Horizontal
        painter.drawLine(size//2, 10, size//2, size - 10)  # Vertical
    
    def get_color_scheme(self):
        """Return the color scheme for consistent styling"""
        return self.color_scheme


# Global icon manager instance
_icon_manager = None

def get_icon_manager(project_root=None):
    """Get the global icon manager instance"""
    global _icon_manager
    if _icon_manager is None:
        if project_root is None:
            # Try to find project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to PyMeshGen/
        
        _icon_manager = IconManager(project_root)
    
    return _icon_manager


def get_icon(icon_name):
    """Get icon by name using the global icon manager"""
    manager = get_icon_manager()
    return manager.get_icon(icon_name)
