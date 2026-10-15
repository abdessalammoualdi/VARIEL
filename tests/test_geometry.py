"""Identities checked against the original source-locked geometry implementation."""
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'variel/frozen/sparse'))
from dtrar_final_core import brier_rows, prospective_H, simplex

class GeometryTests(unittest.TestCase):
    def test_probabilities_simplex(self):
        a=simplex(np.array([[0.8,0.7,-0.1],[0.5,0.5,0.5]]))
        self.assertTrue(np.all(a>=0))
        np.testing.assert_allclose(a.sum(axis=1),1)

    def test_exact_prospective_liability(self):
        anchor=np.array([[0.6,0.4],[0.2,0.8],[0.7,0.3]])
        new=np.array([[0.65,0.35],[0.4,0.6],[0.68,0.32]])
        delta=new-anchor
        brute_per_row=[]
        for i in range(len(anchor)):
            changes=[]
            for y in range(2):
                b=brier_rows(anchor[i:i+1],np.array([y]))[0]
                q=brier_rows(new[i:i+1],np.array([y]))[0]
                changes.append(q-b)
            brute_per_row.append(max(changes))
        self.assertAlmostEqual(float(np.mean(brute_per_row)), prospective_H(anchor,delta),places=12)

    def test_hold_liability(self):
        anchor=np.array([[0.2,0.8],[0.7,0.3]])
        self.assertAlmostEqual(prospective_H(anchor,np.zeros_like(anchor)),0.0)

if __name__=='__main__': unittest.main()
