#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backup of duplicated dialog button styles extracted from GUI dialogs.
These styles are now centralized in gui.ui_utils.UIStyles.
"""

DIALOG_PRIMARY_BUTTON_STYLE_BACKUP = """
    QPushButton {
        background-color: #e6f7ff;
        border: 1px solid #0078d4;
        border-radius: 3px;
        padding: 6px 12px;
        color: #0078d4;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #cceeff;
    }
    QPushButton:pressed {
        background-color: #99ddff;
    }
"""

DIALOG_SECONDARY_BUTTON_STYLE_BACKUP = """
    QPushButton {
        background-color: #f5f5f5;
        border: 1px solid #cccccc;
        border-radius: 3px;
        padding: 6px 12px;
    }
    QPushButton:hover {
        background-color: #e6e6e6;
    }
    QPushButton:pressed {
        background-color: #d9d9d9;
    }
"""
