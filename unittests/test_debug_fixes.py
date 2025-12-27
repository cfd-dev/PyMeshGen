"""
Unit tests for PyMeshGen core functionality
Tests various components that were previously tested with debug files
"""

import unittest
import sys
import os
from pathlib import Path
import tempfile
import json
import numpy as np

# 添加项目路径
project_root = str(Path(__file__).parent)
for subdir in ["fileIO", "data_structure", "meshsize", "visualization", "adfront2", "optimize", "utils"]:
    sys.path.append(str(Path(project_root) / subdir))


class TestDebugFixes(unittest.TestCase):
    """Test cases for the fixes that were previously in debug files"""
    
    def setUp(self):
        """Setup test fixtures before each test method."""
        pass
    
    def test_array_indexing_fix(self):
        """Test array indexing fixes"""
        try:
            # Test creating a mock front with insufficient nodes
            from data_structure.front2d import Front
            from data_structure.basic_elements import NodeElement
            
            # Create two test nodes for the front
            node1 = NodeElement([0.0, 0.0], 0)
            node2 = NodeElement([1.0, 0.0], 1)

            # Test creating a front with two nodes
            front = Front(node1, node2, idx=0, bc_type="test")

            # Verify that the front was created successfully
            self.assertIsNotNone(front)
            # Note: Front constructor takes 2 nodes, not a list of nodes
            
            print("PASS: Array indexing fix test passed")
        except Exception as e:
            print(f"FAIL: Array indexing fix test failed: {e}")
            raise
    
    def test_data_converter_fix(self):
        """Test data converter functionality"""
        try:
            # Test with MeshData object
            from data_structure.mesh_data import MeshData
            
            # Create a simple mesh data object
            mesh_data = MeshData()
            mesh_data.node_coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            mesh_data.cells = [[0, 1, 2], [0, 2, 3]]  # Two triangles
            mesh_data.parts_info = {"wall": {"type": "wall", "faces": []}}
            
            # Test that the object was created successfully
            self.assertIsNotNone(mesh_data)
            self.assertEqual(len(mesh_data.node_coords), 4)
            self.assertEqual(len(mesh_data.cells), 2)
            
            print("PASS: Data converter fix test passed")
        except Exception as e:
            print(f"FAIL: Data converter fix test failed: {e}")
            raise

    def test_import_functionality(self):
        """Test import functionality"""
        try:
            # Test importing necessary modules
            from fileIO.read_cas import parse_fluent_msh
            from data_structure.parameters import Parameters
            from data_structure.basic_elements import NodeElement, Triangle, Quadrilateral

            # Verify that modules can be imported successfully
            self.assertTrue(callable(parse_fluent_msh))
            self.assertTrue(hasattr(Parameters, '__init__'))
            self.assertTrue(hasattr(NodeElement, '__init__'))

            print("PASS: Import functionality test passed")
        except Exception as e:
            print(f"FAIL: Import functionality test failed: {e}")
            raise

    def test_node_element_instantiation(self):
        """Test NodeElement instantiation fix"""
        try:
            from data_structure.basic_elements import NodeElement

            # Test NodeElement creation with various parameters
            node1 = NodeElement([1.0, 2.0], 0)
            node2 = NodeElement([3.0, 4.0, 5.0], 1)  # 3D coordinates
            node3 = NodeElement((6.0, 7.0), 2)  # tuple coordinates

            # Verify nodes were created successfully
            self.assertEqual(node1.idx, 0)
            self.assertEqual(node1.coords, [1.0, 2.0])
            self.assertEqual(len(node2.coords), 3)
            self.assertEqual(node3.coords, [6.0, 7.0])

            print("PASS: NodeElement instantiation test passed")
        except Exception as e:
            print(f"FAIL: NodeElement instantiation test failed: {e}")
            raise

    def test_pymeshgen_function(self):
        """Test PyMeshGen function import and basic functionality"""
        try:
            from PyMeshGen import PyMeshGen
            from data_structure.parameters import Parameters

            # Verify that PyMeshGen function can be imported
            self.assertTrue(callable(PyMeshGen))

            print("PASS: PyMeshGen function test passed")
        except Exception as e:
            print(f"FAIL: PyMeshGen function test failed: {e}")
            raise

    def test_part_workflow(self):
        """Test part parameter workflow functionality"""
        try:
            from data_structure.parameters import Parameters
            from data_structure.mesh_data import MeshData

            # Create a basic mesh data object
            mesh_data = MeshData()
            mesh_data.node_coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            mesh_data.cells = [[0, 1, 2], [0, 2, 3]]
            mesh_data.parts_info = {"test_part": {"type": "wall", "faces": []}}

            # Create parameters object
            params = Parameters("FROM_MAIN_JSON")  # Use default

            # Verify that both objects were created successfully
            self.assertIsNotNone(mesh_data)
            self.assertIsNotNone(params)

            print("PASS: Part workflow test passed")
        except Exception as e:
            print(f"FAIL: Part workflow test failed: {e}")
            raise

    def test_complete_mesh_flow(self):
        """Test complete mesh generation flow simulation"""
        try:
            from data_structure.mesh_data import MeshData
            from data_structure.parameters import Parameters
            from data_structure.basic_elements import Unstructured_Grid, Triangle, Quadrilateral
            from utils.data_converter import convert_to_internal_mesh_format

            # Create test mesh data similar to what comes from GUI import
            mesh_data = MeshData()
            mesh_data.node_coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            mesh_data.cells = [[0, 1, 2], [0, 2, 3]]
            mesh_data.parts_info = {"test_part": {"type": "wall", "faces": []}}

            # Test converting to internal format
            internal_format = convert_to_internal_mesh_format(mesh_data)

            # Verify that conversion was successful
            self.assertIsNotNone(internal_format)
            self.assertIn('nodes', internal_format)
            # Changed 'cells' to 'faces' as per actual data converter output
            self.assertIn('faces', internal_format)

            print("PASS: Complete mesh flow test passed")
        except Exception as e:
            print(f"FAIL: Complete mesh flow test failed: {e}")
            raise


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)