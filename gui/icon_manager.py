"""
PyMeshGen Icon Manager
Provides optimized icons for all GUI functions with consistent styling
"""

import os
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QBrush
from PyQt5.QtCore import Qt
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
            'document-import': QStyle.SP_ArrowUp,
            'document-export': QStyle.SP_ArrowDown,
            'import': QStyle.SP_ArrowUp,
            'export': QStyle.SP_ArrowDown,
            
            # View operations
            'zoom-fit-best': QStyle.SP_ComputerIcon,
            'zoom-in': QStyle.SP_ArrowUp,
            'zoom-out': QStyle.SP_ArrowDown,
            'view-refresh': QStyle.SP_BrowserReload,
            'view-fullscreen': QStyle.SP_TitleBarMaxButton,
            
            # Edit operations
            'edit-delete': QStyle.SP_TrashIcon,
            'edit-clear': QStyle.SP_LineEditClearButton,
            'system-run': QStyle.SP_MediaPlay,
            'run': QStyle.SP_MediaPlay,
            
            # Configuration and settings
            'configure': QStyle.SP_ToolBarHorizontalExtensionButton,
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
                return app.style().standardIcon(icon_map[icon_name])
        
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
        
        # Import required classes
        from PyQt5.QtGui import QPolygon
        from PyQt5.QtCore import QPoint

        # Draw different icons based on icon_name
        if icon_name in ['mesh-generate', 'generate-mesh', 'mesh_generate', 'mesh-generate']:
            self._draw_mesh_generate_icon(painter, size, primary_color)
        elif icon_name in ['mesh-quality', 'check-quality', 'mesh_quality', 'quality']:
            self._draw_mesh_quality_icon(painter, size, secondary_color)
        elif icon_name in ['mesh-smooth', 'smooth-mesh', 'mesh_smooth', 'smooth']:
            self._draw_mesh_smooth_icon(painter, size, primary_color)
        elif icon_name in ['mesh-optimize', 'optimize-mesh', 'mesh_optimize', 'optimize']:
            self._draw_mesh_optimize_icon(painter, size, accent_color)
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
        else:
            # Default fallback: draw a generic icon
            self._draw_generic_icon(painter, size, primary_color)
        
        painter.end()
        
        return QIcon(pixmap)
    
    def _draw_mesh_generate_icon(self, painter, size, color):
        """Draw mesh generation icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Draw a grid pattern
        cell_size = size // 4
        for i in range(5):
            # Horizontal lines
            painter.drawLine(0, i * cell_size, size, i * cell_size)
            # Vertical lines
            painter.drawLine(i * cell_size, 0, i * cell_size, size)
        
        # Draw play button in center
        from PyQt5.QtGui import QPolygon
        from PyQt5.QtCore import QPoint
        painter.setBrush(color)
        points = [
            (size//2 - 5, size//2 - 8),
            (size//2 + 7, size//2),
            (size//2 - 5, size//2 + 8)
        ]
        # Convert list of tuples to QPolygon
        qpoints = [QPoint(int(x), int(y)) for x, y in points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
    
    def _draw_mesh_quality_icon(self, painter, size, color):
        """Draw mesh quality icon"""
        from PyQt5.QtGui import QPolygon
        from PyQt5.QtCore import QPoint
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
        """Draw mesh smoothing icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Draw wavy line representing smoothing
        center_y = size // 2
        amplitude = size // 6
        
        # Draw a smooth sine wave
        points = []
        for x in range(0, size, 2):
            y = center_y + amplitude * 0.7 * (x / size) * (size - x) / size  # Parabolic curve
            points.append((x, int(y)))
        
        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
    
    def _draw_mesh_optimize_icon(self, painter, size, color):
        """Draw mesh optimization icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Draw an upward arrow with optimization symbol
        # Arrow shaft
        painter.drawLine(size//2, size - 5, size//2, 10)
        # Arrow head
        painter.drawLine(size//2, 10, size//2 - 5, 18)
        painter.drawLine(size//2, 10, size//2 + 5, 18)
        
        # Draw optimization symbol (gear)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(size//2 - 8, size//2 + 5, 16, 16)
        
        # Draw gear teeth
        painter.setBrush(color)
        for i in range(6):
            painter.save()
            painter.translate(size//2, size//2 + 5)
            painter.rotate(i * 60)
            painter.drawRect(-2, -10, 4, 4)  # Small rectangle as gear tooth
            painter.restore()
    
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
        
        # Draw export arrow (upward arrow)
        painter.drawLine(size - 10, size - 10, size - 10, size//2 + 5)
        painter.drawLine(size - 10, size//2 + 5, size - 15, size//2 + 10)
        painter.drawLine(size - 10, size//2 + 5, size - 5, size//2 + 10)
    
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
        from PyQt5.QtGui import QPolygon
        from PyQt5.QtCore import QPoint
        
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
        """Draw X+ view icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Draw 3D coordinate system box
        margin = 5
        box_size = size - 2 * margin
        
        # Draw back face (darker)
        back_pen = QPen(QColor('#CCCCCC'), 1)
        painter.setPen(back_pen)
        painter.drawRect(margin, margin, box_size, box_size)
        
        # Draw front face with X+ direction highlighted
        painter.setPen(pen)
        front_offset = box_size // 3
        painter.drawRect(margin + front_offset, margin + front_offset, box_size, box_size)
        
        # Draw X+ arrow
        painter.setBrush(color)
        arrow_x = size - 8
        arrow_y = size // 2
        
        # Arrow shaft
        painter.drawLine(arrow_x - 6, arrow_y, arrow_x, arrow_y)
        # Arrow head
        from PyQt5.QtGui import QPolygon
        from PyQt5.QtCore import QPoint
        arrow_points = [
            (arrow_x, arrow_y),
            (arrow_x - 4, arrow_y - 4),
            (arrow_x - 4, arrow_y + 4)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in arrow_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw X label
        painter.setPen(color)
        painter.drawText(arrow_x - 6, arrow_y - 8, "X")
    
    def _draw_view_x_neg_icon(self, painter, size, color):
        """Draw X- view icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Draw 3D coordinate system box
        margin = 5
        box_size = size - 2 * margin
        
        # Draw back face (darker)
        back_pen = QPen(QColor('#CCCCCC'), 1)
        painter.setPen(back_pen)
        painter.drawRect(margin, margin, box_size, box_size)
        
        # Draw front face with X- direction highlighted
        painter.setPen(pen)
        front_offset = box_size // 3
        painter.drawRect(margin - front_offset, margin + front_offset, box_size, box_size)
        
        # Draw X- arrow
        painter.setBrush(color)
        arrow_x = 8
        arrow_y = size // 2
        
        # Arrow shaft
        painter.drawLine(arrow_x + 6, arrow_y, arrow_x, arrow_y)
        # Arrow head
        from PyQt5.QtGui import QPolygon
        from PyQt5.QtCore import QPoint
        arrow_points = [
            (arrow_x, arrow_y),
            (arrow_x + 4, arrow_y - 4),
            (arrow_x + 4, arrow_y + 4)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in arrow_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw X label
        painter.setPen(color)
        painter.drawText(arrow_x + 2, arrow_y - 8, "X")
    
    def _draw_view_y_pos_icon(self, painter, size, color):
        """Draw Y+ view icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Draw 3D coordinate system box
        margin = 5
        box_size = size - 2 * margin
        
        # Draw back face (darker)
        back_pen = QPen(QColor('#CCCCCC'), 1)
        painter.setPen(back_pen)
        painter.drawRect(margin, margin, box_size, box_size)
        
        # Draw front face with Y+ direction highlighted
        painter.setPen(pen)
        front_offset = box_size // 3
        painter.drawRect(margin + front_offset, margin - front_offset, box_size, box_size)
        
        # Draw Y+ arrow
        painter.setBrush(color)
        arrow_x = size // 2
        arrow_y = 8
        
        # Arrow shaft
        painter.drawLine(arrow_x, arrow_y + 6, arrow_x, arrow_y)
        # Arrow head
        from PyQt5.QtGui import QPolygon
        from PyQt5.QtCore import QPoint
        arrow_points = [
            (arrow_x, arrow_y),
            (arrow_x - 4, arrow_y + 4),
            (arrow_x + 4, arrow_y + 4)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in arrow_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw Y label
        painter.setPen(color)
        painter.drawText(arrow_x + 6, arrow_y + 8, "Y")
    
    def _draw_view_y_neg_icon(self, painter, size, color):
        """Draw Y- view icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Draw 3D coordinate system box
        margin = 5
        box_size = size - 2 * margin
        
        # Draw back face (darker)
        back_pen = QPen(QColor('#CCCCCC'), 1)
        painter.setPen(back_pen)
        painter.drawRect(margin, margin, box_size, box_size)
        
        # Draw front face with Y- direction highlighted
        painter.setPen(pen)
        front_offset = box_size // 3
        painter.drawRect(margin + front_offset, margin + front_offset, box_size, box_size)
        
        # Draw Y- arrow
        painter.setBrush(color)
        arrow_x = size // 2
        arrow_y = size - 8
        
        # Arrow shaft
        painter.drawLine(arrow_x, arrow_y - 6, arrow_x, arrow_y)
        # Arrow head
        from PyQt5.QtGui import QPolygon
        from PyQt5.QtCore import QPoint
        arrow_points = [
            (arrow_x, arrow_y),
            (arrow_x - 4, arrow_y - 4),
            (arrow_x + 4, arrow_y - 4)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in arrow_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw Y label
        painter.setPen(color)
        painter.drawText(arrow_x + 6, arrow_y - 2, "Y")
    
    def _draw_view_z_pos_icon(self, painter, size, color):
        """Draw Z+ view icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Draw 3D coordinate system box
        margin = 5
        box_size = size - 2 * margin
        
        # Draw back face (darker)
        back_pen = QPen(QColor('#CCCCCC'), 1)
        painter.setPen(back_pen)
        painter.drawRect(margin, margin, box_size, box_size)
        
        # Draw front face with Z+ direction highlighted
        painter.setPen(pen)
        front_offset = box_size // 3
        painter.drawRect(margin + front_offset, margin + front_offset, box_size, box_size)
        
        # Draw Z+ arrow (pointing out of screen)
        painter.setBrush(color)
        arrow_x = size // 2
        arrow_y = size // 2
        
        # Draw circle with dot for Z+ (pointing out)
        painter.drawEllipse(arrow_x - 6, arrow_y - 6, 12, 12)
        painter.drawEllipse(arrow_x - 2, arrow_y - 2, 4, 4)
        
        # Draw Z label
        painter.setPen(color)
        painter.drawText(arrow_x + 8, arrow_y + 4, "Z")
    
    def _draw_view_z_neg_icon(self, painter, size, color):
        """Draw Z- view icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Draw 3D coordinate system box
        margin = 5
        box_size = size - 2 * margin
        
        # Draw back face (darker)
        back_pen = QPen(QColor('#CCCCCC'), 1)
        painter.setPen(back_pen)
        painter.drawRect(margin, margin, box_size, box_size)
        
        # Draw front face with Z- direction highlighted
        painter.setPen(pen)
        front_offset = box_size // 3
        painter.drawRect(margin - front_offset, margin - front_offset, box_size, box_size)
        
        # Draw Z- arrow (pointing into screen)
        painter.setBrush(color)
        arrow_x = size // 2
        arrow_y = size // 2
        
        # Draw circle with X for Z- (pointing in)
        painter.drawEllipse(arrow_x - 6, arrow_y - 6, 12, 12)
        painter.drawLine(arrow_x - 4, arrow_y - 4, arrow_x + 4, arrow_y + 4)
        painter.drawLine(arrow_x + 4, arrow_y - 4, arrow_x - 4, arrow_y + 4)
        
        # Draw Z label
        painter.setPen(color)
        painter.drawText(arrow_x + 8, arrow_y + 4, "Z")
    
    def _draw_view_iso_icon(self, painter, size, color):
        """Draw isometric view icon"""
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Draw isometric cube
        center_x = size // 2
        center_y = size // 2
        cube_size = size // 4
        
        # Draw top face
        from PyQt5.QtGui import QPolygon
        from PyQt5.QtCore import QPoint
        
        top_points = [
            (center_x, center_y - cube_size),
            (center_x + cube_size * 0.866, center_y - cube_size * 0.5),
            (center_x, center_y),
            (center_x - cube_size * 0.866, center_y - cube_size * 0.5)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in top_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw left face
        left_points = [
            (center_x - cube_size * 0.866, center_y - cube_size * 0.5),
            (center_x, center_y),
            (center_x, center_y + cube_size),
            (center_x - cube_size * 0.866, center_y + cube_size * 0.5)
        ]
        qpoints = [QPoint(int(x), int(y)) for x, y in left_points]
        qpolygon = QPolygon(qpoints)
        painter.drawPolygon(qpolygon)
        
        # Draw right face
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