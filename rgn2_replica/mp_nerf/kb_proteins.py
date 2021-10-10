# Author: Eric Alcaide

# A substantial part has been borrowed from
# https://github.com/jonathanking/sidechainnet
#
# Here's the License for it:
#
# Copyright 2020 Jonathan King
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np

#########################
### FROM SIDECHAINNET ###
#########################

# modified by considering rigid bodies in sidechains (remove extra torsions)

SC_BUILD_INFO = {
    'A': {
        'angles-names': ['N-CA-CB'],
        'angles-types': ['N -CX-CT'],
        'angles-vals': [1.9146261894377796],
        'atom-names': ['CB'],
        'bonds-names': ['CA-CB'],
        'bonds-types': ['CX-CT'],
        'bonds-vals': [1.526],
        'torsion-names': ['C-N-CA-CB'],
        'torsion-types': ['C -N -CX-CT'],
        'torsion-vals': ['p'],
        'rigid-frames-idxs': [[0,1,2], [0,1,4]],
    },

    'R': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-NE', 'CD-NE-CZ', 'NE-CZ-NH1',
            'NE-CZ-NH2'
        ],
        'angles-types': [
            'N -CX-C8', 'CX-C8-C8', 'C8-C8-C8', 'C8-C8-N2', 'C8-N2-CA', 'N2-CA-N2',
            'N2-CA-N2'
        ],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.9408061282176945,
            2.150245638457014, 2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-NE', 'NE-CZ', 'CZ-NH1', 'CZ-NH2'],
        'bonds-types': ['CX-C8', 'C8-C8', 'C8-C8', 'C8-N2', 'N2-CA', 'CA-N2', 'CA-N2'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.463, 1.34, 1.34, 1.34],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-NE', 'CG-CD-NE-CZ',
            'CD-NE-CZ-NH1', 'CD-NE-CZ-NH2'
        ],
        'torsion-types': [
            'C -N -CX-C8', 'N -CX-C8-C8', 'CX-C8-C8-C8', 'C8-C8-C8-N2', 'C8-C8-N2-CA',
            'C8-N2-CA-N2', 'C8-N2-CA-N2'
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'p', 0., 3.141592],
        'rigid-frames-idxs': [[0,1,2], [0,1,4], [1,4,5], [4,5,6], [5,6,7], [6,7,8]],
    },

    'N': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-OD1', 'CB-CG-ND2'],
        'angles-types': ['N -CX-2C', 'CX-2C-C ', '2C-C -O ', '2C-C -N '],
        'angles-vals': [
            1.9146261894377796, 1.9390607989657, 2.101376419401173, 2.035053907825388
        ],
        'atom-names': ['CB', 'CG', 'OD1', 'ND2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-OD1', 'CG-ND2'],
        'bonds-types': ['CX-2C', '2C-C ', 'C -O ', 'C -N '],
        'bonds-vals': [1.526, 1.522, 1.229, 1.335],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-OD1', 'CA-CB-CG-ND2'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-C ', 'CX-2C-C -O ', 'CX-2C-C -N '],
        'torsion-vals': ['p', 'p', 'p', 'i'], 
        'rigid-frames-idxs': [[0,1,2], [0,1,4], [1,4,5], [4,5,6]],
    },

    'D': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-OD1', 'CB-CG-OD2'],
        'angles-types': ['N -CX-2C', 'CX-2C-CO', '2C-CO-O2', '2C-CO-O2'],
        'angles-vals': [
            1.9146261894377796, 1.9390607989657, 2.0420352248333655, 2.0420352248333655
        ],
        'atom-names': ['CB', 'CG', 'OD1', 'OD2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-OD1', 'CG-OD2'],
        'bonds-types': ['CX-2C', '2C-CO', 'CO-O2', 'CO-O2'],
        'bonds-vals': [1.526, 1.522, 1.25, 1.25],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-OD1', 'CA-CB-CG-OD2'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-CO', 'CX-2C-CO-O2', 'CX-2C-CO-O2'],
        'torsion-vals': ['p', 'p', 'p', 'i'],
        'rigid-frames-idxs': [[0,1,2], [0,1,4], [1,4,5], [4,5,6]],
    },

    'C': {
        'angles-names': ['N-CA-CB', 'CA-CB-SG'],
        'angles-types': ['N -CX-2C', 'CX-2C-SH'],
        'angles-vals': [1.9146261894377796, 1.8954275676658419],
        'atom-names': ['CB', 'SG'],
        'bonds-names': ['CA-CB', 'CB-SG'],
        'bonds-types': ['CX-2C', '2C-SH'],
        'bonds-vals': [1.526, 1.81],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-SG'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-SH'],
        'torsion-vals': ['p', 'p'],
        'rigid-frames-idxs': [[0,1,2], [0,1,4], [1,4,5]],
    },

    'Q': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-OE1', 'CG-CD-NE2'],
        'angles-types': ['N -CX-2C', 'CX-2C-2C', '2C-2C-C ', '2C-C -O ', '2C-C -N '],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.9390607989657, 2.101376419401173,
            2.035053907825388
        ],
        'atom-names': ['CB', 'CG', 'CD', 'OE1', 'NE2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-OE1', 'CD-NE2'],
        'bonds-types': ['CX-2C', '2C-2C', '2C-C ', 'C -O ', 'C -N '],
        'bonds-vals': [1.526, 1.526, 1.522, 1.229, 1.335],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-OE1', 'CB-CG-CD-NE2'
        ],
        'torsion-types': [
            'C -N -CX-2C', 'N -CX-2C-2C', 'CX-2C-2C-C ', '2C-2C-C -O ', '2C-2C-C -N '
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'i'],
        'rigid-frames-idxs': [[0,1,2], [0,1,4], [1,4,5], [4,5,6], [5,6,7]],
    },

    'E': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-OE1', 'CG-CD-OE2'],
        'angles-types': ['N -CX-2C', 'CX-2C-2C', '2C-2C-CO', '2C-CO-O2', '2C-CO-O2'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.9390607989657, 2.0420352248333655,
            2.0420352248333655
        ],
        'atom-names': ['CB', 'CG', 'CD', 'OE1', 'OE2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-OE1', 'CD-OE2'],
        'bonds-types': ['CX-2C', '2C-2C', '2C-CO', 'CO-O2', 'CO-O2'],
        'bonds-vals': [1.526, 1.526, 1.522, 1.25, 1.25],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-OE1', 'CB-CG-CD-OE2'
        ],
        'torsion-types': [
            'C -N -CX-2C', 'N -CX-2C-2C', 'CX-2C-2C-CO', '2C-2C-CO-O2', '2C-2C-CO-O2'
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'i'],
        'rigid-frames-idxs': [[0,1,2], [0,1,4], [1,4,5], [4,5,6], [5,6,7]],
    },

    'G': {
        'angles-names': [],
        'angles-types': [],
        'angles-vals': [],
        'atom-names': [],
        'bonds-names': [],
        'bonds-types': [],
        'bonds-vals': [],
        'torsion-names': [],
        'torsion-types': [],
        'torsion-vals': [],
        'rigid-frames-idxs': [[0,1,2]],
    },

    'H': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-ND1', 'CG-ND1-CE1', 'ND1-CE1-NE2', 'CE1-NE2-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-CC', 'CT-CC-NA', 'CC-NA-CR', 'NA-CR-NB', 'CR-NB-CV'
        ],
        'angles-vals': [
            1.9146261894377796, 1.9739673840055867, 2.0943951023931953,
            1.8849555921538759, 1.8849555921538759, 1.8849555921538759
        ],
        'atom-names': ['CB', 'CG', 'ND1', 'CE1', 'NE2', 'CD2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-ND1', 'ND1-CE1', 'CE1-NE2', 'NE2-CD2'],
        'bonds-types': ['CX-CT', 'CT-CC', 'CC-NA', 'NA-CR', 'CR-NB', 'NB-CV'],
        'bonds-vals': [1.526, 1.504, 1.385, 1.343, 1.335, 1.394],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-ND1', 'CB-CG-ND1-CE1', 'CG-ND1-CE1-NE2',
            'ND1-CE1-NE2-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-CC', 'CX-CT-CC-NA', 'CT-CC-NA-CR', 'CC-NA-CR-NB',
            'NA-CR-NB-CV'
        ],
        'torsion-vals': ['p', 'p', 'p', 3.141592653589793, 0.0, 0.0],
        'rigid-frames-idxs': [[0,1,2], [0,1,4], [1,4,5], [4,5,6]],
    },

    'I': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG1', 'CB-CG1-CD1', 'CA-CB-CG2'],
        'angles-types': ['N -CX-3C', 'CX-3C-2C', '3C-2C-CT', 'CX-3C-CT'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.911135530933791
        ],
        'atom-names': ['CB', 'CG1', 'CD1', 'CG2'],
        'bonds-names': ['CA-CB', 'CB-CG1', 'CG1-CD1', 'CB-CG2'],
        'bonds-types': ['CX-3C', '3C-2C', '2C-CT', '3C-CT'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG1', 'CA-CB-CG1-CD1', 'N-CA-CB-CG2'],
        'torsion-types': ['C -N -CX-3C', 'N -CX-3C-2C', 'CX-3C-2C-CT', 'N -CX-3C-CT'],
        'torsion-vals': ['p', 'p', 'p', -2.1315], # last one was 'p' in the original - but cg1-cg2 = "2.133"
        'rigid-frames-idxs': [[0,1,2], [0,1,4], [1,4,5], [4,5,7]],
    },

    'L': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CB-CG-CD2'],
        'angles-types': ['N -CX-2C', 'CX-2C-3C', '2C-3C-CT', '2C-3C-CT'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.911135530933791
        ],
        'atom-names': ['CB', 'CG', 'CD1', 'CD2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD1', 'CG-CD2'],
        'bonds-types': ['CX-2C', '2C-3C', '3C-CT', '3C-CT'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CA-CB-CG-CD2'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-3C', 'CX-2C-3C-CT', 'CX-2C-3C-CT'],
        # extra torsion is in negative bc in mask construction, previous angle is summed. 
        'torsion-vals': ['p', 'p', 'p', 2.1315], # last one was 'p' in the original - but cd1-cd2 = "-2.130"
        'rigid-frames-idxs': [[0,1,2], [0,1,4], [1,4,5], [4,5,6]],
    },

    'K': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-CE', 'CD-CE-NZ'],
        'angles-types': ['N -CX-C8', 'CX-C8-C8', 'C8-C8-C8', 'C8-C8-C8', 'C8-C8-N3'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.911135530933791,
            1.9408061282176945
        ],
        'atom-names': ['CB', 'CG', 'CD', 'CE', 'NZ'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-CE', 'CE-NZ'],
        'bonds-types': ['CX-C8', 'C8-C8', 'C8-C8', 'C8-C8', 'C8-N3'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.526, 1.471],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-CE', 'CG-CD-CE-NZ'
        ],
        'torsion-types': [
            'C -N -CX-C8', 'N -CX-C8-C8', 'CX-C8-C8-C8', 'C8-C8-C8-C8', 'C8-C8-C8-N3'
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'p'],
        'rigid-frames-idxs': [[0,1,2], [0,1,4], [1,4,5], [4,5,6], [5,6,7], [6,7,8]],
    },

    'M': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-SD', 'CG-SD-CE'],
        'angles-types': ['N -CX-2C', 'CX-2C-2C', '2C-2C-S ', '2C-S -CT'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 2.0018926520374962, 1.726130630222392
        ],
        'atom-names': ['CB', 'CG', 'SD', 'CE'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-SD', 'SD-CE'],
        'bonds-types': ['CX-2C', '2C-2C', '2C-S ', 'S -CT'],
        'bonds-vals': [1.526, 1.526, 1.81, 1.81],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-SD', 'CB-CG-SD-CE'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-2C', 'CX-2C-2C-S ', '2C-2C-S -CT'],
        'torsion-vals': ['p', 'p', 'p', 'p'],
        'rigid-frames-idxs': [[0,1,2], [0,1,4], [1,4,5], [4,5,6], [5,6,7]],
    },

    'F': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CG-CD1-CE1', 'CD1-CE1-CZ', 'CE1-CZ-CE2',
            'CZ-CE2-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-CA', 'CT-CA-CA', 'CA-CA-CA', 'CA-CA-CA', 'CA-CA-CA',
            'CA-CA-CA'
        ],
        'angles-vals': [
            1.9146261894377796, 1.9896753472735358, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': ['CB', 'CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2'],
        'bonds-names': [
            'CA-CB', 'CB-CG', 'CG-CD1', 'CD1-CE1', 'CE1-CZ', 'CZ-CE2', 'CE2-CD2'
        ],
        'bonds-types': ['CX-CT', 'CT-CA', 'CA-CA', 'CA-CA', 'CA-CA', 'CA-CA', 'CA-CA'],
        'bonds-vals': [1.526, 1.51, 1.4, 1.4, 1.4, 1.4, 1.4],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CB-CG-CD1-CE1', 'CG-CD1-CE1-CZ',
            'CD1-CE1-CZ-CE2', 'CE1-CZ-CE2-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-CA', 'CX-CT-CA-CA', 'CT-CA-CA-CA', 'CA-CA-CA-CA',
            'CA-CA-CA-CA', 'CA-CA-CA-CA'
        ],
        'torsion-vals': ['p', 'p', 'p', 3.141592653589793, 0.0, 0.0, 0.0],
        'rigid-frames-idxs': [[0,1,2], [0,1,4], [1,4,5], [4,5,6]],
    },

    'P': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD'],
        'angles-types': ['N -CX-CT', 'CX-CT-CT', 'CT-CT-CT'],
        'angles-vals': [1.9146261894377796, 1.911135530933791, 1.911135530933791],
        'atom-names': ['CB', 'CG', 'CD'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD'],
        'bonds-types': ['CX-CT', 'CT-CT', 'CT-CT'],
        'bonds-vals': [1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD'],
        'torsion-types': ['C -N -CX-CT', 'N -CX-CT-CT', 'CX-CT-CT-CT'],
        'torsion-vals': ['p', 'p', 'p'],
        'rigid-frames-idxs': [[0,1,2], [0,1,4], [1,4,5], [4,5,6]],
    },

    'S': {
        'angles-names': ['N-CA-CB', 'CA-CB-OG'],
        'angles-types': ['N -CX-2C', 'CX-2C-OH'],
        'angles-vals': [1.9146261894377796, 1.911135530933791],
        'atom-names': ['CB', 'OG'],
        'bonds-names': ['CA-CB', 'CB-OG'],
        'bonds-types': ['CX-2C', '2C-OH'],
        'bonds-vals': [1.526, 1.41],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-OG'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-OH'],
        'torsion-vals': ['p', 'p'],
        'rigid-frames-idxs': [[0,1,2], [0,1,4], [1,4,5]],
    },

    'T': {
        'angles-names': ['N-CA-CB', 'CA-CB-OG1', 'CA-CB-CG2'],
        'angles-types': ['N -CX-3C', 'CX-3C-OH', 'CX-3C-CT'],
        'angles-vals': [1.9146261894377796, 1.911135530933791, 1.911135530933791],
        'atom-names': ['CB', 'OG1', 'CG2'],
        'bonds-names': ['CA-CB', 'CB-OG1', 'CB-CG2'],
        'bonds-types': ['CX-3C', '3C-OH', '3C-CT'],
        'bonds-vals': [1.526, 1.41, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-OG1', 'N-CA-CB-CG2'],
        'torsion-types': ['C -N -CX-3C', 'N -CX-3C-OH', 'N -CX-3C-CT'],
        'torsion-vals': ['p', 'p', 'p'],
        'rigid-frames-idxs': [[0,1,2], [0,1,4], [1,4,5]],
    },

    'W': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CG-CD1-NE1', 'CD1-NE1-CE2',
            'NE1-CE2-CZ2', 'CE2-CZ2-CH2', 'CZ2-CH2-CZ3', 'CH2-CZ3-CE3', 'CZ3-CE3-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-C*', 'CT-C*-CW', 'C*-CW-NA', 'CW-NA-CN', 'NA-CN-CA',
            'CN-CA-CA', 'CA-CA-CA', 'CA-CA-CA', 'CA-CA-CB'
        ],
        'angles-vals': [
            1.9146261894377796, 2.0176006153054447, 2.181661564992912, 1.8971728969178363,
            1.9477874452256716, 2.3177972466484698, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': [
            'CB', 'CG', 'CD1', 'NE1', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3', 'CD2'
        ],
        'bonds-names': [
            'CA-CB', 'CB-CG', 'CG-CD1', 'CD1-NE1', 'NE1-CE2', 'CE2-CZ2', 'CZ2-CH2',
            'CH2-CZ3', 'CZ3-CE3', 'CE3-CD2'
        ],
        'bonds-types': [
            'CX-CT', 'CT-C*', 'C*-CW', 'CW-NA', 'NA-CN', 'CN-CA', 'CA-CA', 'CA-CA',
            'CA-CA', 'CA-CB'
        ],
        'bonds-vals': [1.526, 1.495, 1.352, 1.381, 1.38, 1.4, 1.4, 1.4, 1.4, 1.404],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CB-CG-CD1-NE1', 'CG-CD1-NE1-CE2',
            'CD1-NE1-CE2-CZ2', 'NE1-CE2-CZ2-CH2', 'CE2-CZ2-CH2-CZ3', 'CZ2-CH2-CZ3-CE3',
            'CH2-CZ3-CE3-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-C*', 'CX-CT-C*-CW', 'CT-C*-CW-NA', 'C*-CW-NA-CN',
            'CW-NA-CN-CA', 'NA-CN-CA-CA', 'CN-CA-CA-CA', 'CA-CA-CA-CA', 'CA-CA-CA-CB'
        ],
        'torsion-vals': [
            'p', 'p', 'p', 3.141592653589793, 0.0, 3.141592653589793, 3.141592653589793,
            0.0, 0.0, 0.0
        ],
        'rigid-frames-idxs': [[0,1,2], [0,1,4], [1,4,5], [4,5,6]]
    },

    'Y': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CG-CD1-CE1', 'CD1-CE1-CZ', 'CE1-CZ-OH',
            'CE1-CZ-CE2', 'CZ-CE2-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-CA', 'CT-CA-CA', 'CA-CA-CA', 'CA-CA-C ', 'CA-C -OH',
            'CA-C -CA', 'C -CA-CA'
        ],
        'angles-vals': [
            1.9146261894377796, 1.9896753472735358, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': ['CB', 'CG', 'CD1', 'CE1', 'CZ', 'OH', 'CE2', 'CD2'],
        'bonds-names': [
            'CA-CB', 'CB-CG', 'CG-CD1', 'CD1-CE1', 'CE1-CZ', 'CZ-OH', 'CZ-CE2', 'CE2-CD2'
        ],
        'bonds-types': [
            'CX-CT', 'CT-CA', 'CA-CA', 'CA-CA', 'CA-C ', 'C -OH', 'C -CA', 'CA-CA'
        ],
        'bonds-vals': [1.526, 1.51, 1.4, 1.4, 1.409, 1.364, 1.409, 1.4],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CB-CG-CD1-CE1', 'CG-CD1-CE1-CZ',
            'CD1-CE1-CZ-OH', 'CD1-CE1-CZ-CE2', 'CE1-CZ-CE2-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-CA', 'CX-CT-CA-CA', 'CT-CA-CA-CA', 'CA-CA-CA-C ',
            'CA-CA-C -OH', 'CA-CA-C -CA', 'CA-C -CA-CA'
        ],
        'torsion-vals': [
            'p', 'p', 'p', 3.141592653589793, 0.0, 3.141592653589793, 0.0, 0.0
        ],
        'rigid-frames-idxs': [[0,1,2], [0,1,4], [1,4,5], [4,5,6]],
    },

    'V': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG1', 'CA-CB-CG2'],
        'angles-types': ['N -CX-3C', 'CX-3C-CT', 'CX-3C-CT'],
        'angles-vals': [1.9146261894377796, 1.911135530933791, 1.911135530933791],
        'atom-names': ['CB', 'CG1', 'CG2'],
        'bonds-names': ['CA-CB', 'CB-CG1', 'CB-CG2'],
        'bonds-types': ['CX-3C', '3C-CT', '3C-CT'],
        'bonds-vals': [1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG1', 'N-CA-CB-CG2'],
        'torsion-types': ['C -N -CX-3C', 'N -CX-3C-CT', 'N -CX-3C-CT'],
        'torsion-vals': ['p', 'p', 'p'],
        'rigid-frames-idxs': [[0,1,2], [0,1,4], [1,4,5]]
    },

    '_': {
        'angles-names': [],
        'angles-types': [],
        'angles-vals': [],
        'atom-names': [],
        'bonds-names': [],
        'bonds-types': [],
        'bonds-vals': [],
        'torsion-names': [],
        'torsion-types': [],
        'torsion-vals': [],
        'rigid-frames-idxs': [[]],
    }
}

BB_BUILD_INFO = {
    "BONDLENS": {
        # the updated is according to crystal data from 1DPE_1_A and validated with other structures
        # the commented is the sidechainnet one
        'n-ca': 1.4664931, # 1.442, 
        'ca-c': 1.524119,  # 1.498,
        'c-n': 1.3289373,  # 1.379,
        'c-o': 1.229,  # From parm10.dat || huge variability according to structures
        # we get 1.3389416 from 1DPE_1_A but also 1.2289 from 2F2H_d2f2hf1
        'c-oh': 1.364
    },
      # From parm10.dat, for OXT
    # For placing oxygens
    "BONDANGS": {
        'ca-c-o': 2.0944,  # Approximated to be 2pi / 3; parm10.dat says 2.0350539
        'ca-c-oh': 2.0944, 
        'ca-c-n': 2.03,
        'n-ca-c': 1.94,
        'c-n-ca': 2.08,
    },
      # Equal to 'ca-c-o', for OXT
    "BONDTORSIONS": {
        'n-ca-c-n': -0.785398163, # psi (-44 deg, bimodal distro, pick one)
        'c-n-ca-c': -1.3962634015954636, # phi (-80 deg, bimodal distro, pick one)
        'ca-n-c-ca': 3.141592, # omega (180 deg - https://doi.org/10.1016/j.jmb.2005.01.065) 
        'n-ca-c-o': -2.406 # oxygen
    }  # A simple approximation, not meant to be exact.
}


# numbers follow the same order as sidechainnet atoms
SCN_CONNECT = { 
    'A': {
        'bonds': [[0,1], [1,2], [2,3], [1,4]] 
         },
    'R': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [6,7], [7,8], [8,9], [8,10]] 
         },
    'N': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [5,7]] 
         },
    'D': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [5,7]] 
         },
    'C': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5]] 
        },
    'Q': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [6,7], [6,8]] 
        },
    'E': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [6,7], [6,8]]
        },
    'G': {
        'bonds': [[0,1], [1,2], [2,3]] 
        },
    'H': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [6,7], [7,8], [8,9], [5,9]] 
        },
    'I': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [4,7]] 
         },
    'L': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [5,7]] 
         },
    'K': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [6,7], [7,8]] 
         },
    'M': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [6,7]] 
         },
    'F': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [6,7], [7,8], [8,9], [9,10], [5,10]] 
         },
    'P': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [0,6]] 
         },
    'S': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5]] 
         },
    'T': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [4,6]] 
         },
    'W': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [6,7], [7,8], [8,9], [9,10], [10,11], [11,12],
                  [12, 13], [5,13], [8,13]] 
         },
    'Y': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [6,7], [7,8], [8,9], [8,10], [10,11], [5,11]] 
         },
    'V': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [4,6]] 
         },
    '_': {
        'bonds': []
        }
    }

# from: https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf
AMBIGUOUS = {
    "D": {"names": [["OD1", "OD2"]], 
          "indexs": [[6, 7]], 
          }, 
    "E": {"names": [["OE1", "OE2"]],
          "indexs": [[7, 8]], 
          }, 
    "F": {"names": [["CD1", "CD2"], ["CE1", "CE2"]], 
          "indexs": [[6, 10], [7, 9]],
          },
    "Y": {"names": [["CD1", "CD2"], ["CE1", "CE2"]], 
          "indexs": [[6,10], [7,9]],
          },
}


# AA subst mat
BLOSUM = {
    "A" : [4.0, -1.0, -2.0, -2.0, 0.0, -1.0, -1.0, 0.0, -2.0, -1.0, -1.0, -1.0, -1.0, -2.0, -1.0, 1.0, 0.0, -3.0, -2.0, 0.0, 0.0],
    "C" : [-1.0, 5.0, 0.0, -2.0, -3.0, 1.0, 0.0, -2.0, 0.0, -3.0, -2.0, 2.0, -1.0, -3.0, -2.0, -1.0, -1.0, -3.0, -2.0, -3.0, 0.0],
    "D" : [-2.0, 0.0, 6.0, 1.0, -3.0, 0.0, 0.0, 0.0, 1.0, -3.0, -3.0, 0.0, -2.0, -3.0, -2.0, 1.0, 0.0, -4.0, -2.0, -3.0, 0.0],
    "E" : [-2.0, -2.0, 1.0, 6.0, -3.0, 0.0, 2.0, -1.0, -1.0, -3.0, -4.0, -1.0, -3.0, -3.0, -1.0, 0.0, -1.0, -4.0, -3.0, -3.0, 0.0],
    "F" : [0.0, -3.0, -3.0, -3.0, 9.0, -3.0, -4.0, -3.0, -3.0, -1.0, -1.0, -3.0, -1.0, -2.0, -3.0, -1.0, -1.0, -2.0, -2.0, -1.0, 0.0],
    "G" : [-1.0, 1.0, 0.0, 0.0, -3.0, 5.0, 2.0, -2.0, 0.0, -3.0, -2.0, 1.0, 0.0, -3.0, -1.0, 0.0, -1.0, -2.0, -1.0, -2.0, 0.0],
    "H" : [-1.0, 0.0, 0.0, 2.0, -4.0, 2.0, 5.0, -2.0, 0.0, -3.0, -3.0, 1.0, -2.0, -3.0, -1.0, 0.0, -1.0, -3.0, -2.0, -2.0, 0.0],
    "I" : [0.0, -2.0, 0.0, -1.0, -3.0, -2.0, -2.0, 6.0, -2.0, -4.0, -4.0, -2.0, -3.0, -3.0, -2.0, 0.0, -2.0, -2.0, -3.0, -3.0, 0.0],
    "K" : [-2.0, 0.0, 1.0, -1.0, -3.0, 0.0, 0.0, -2.0, 8.0, -3.0, -3.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0, -2.0, 2.0, -3.0, 0.0],
    "L" : [-1.0, -3.0, -3.0, -3.0, -1.0, -3.0, -3.0, -4.0, -3.0, 4.0, 2.0, -3.0, 1.0, 0.0, -3.0, -2.0, -1.0, -3.0, -1.0, 3.0, 0.0],
    "M" : [-1.0, -2.0, -3.0, -4.0, -1.0, -2.0, -3.0, -4.0, -3.0, 2.0, 4.0, -2.0, 2.0, 0.0, -3.0, -2.0, -1.0, -2.0, -1.0, 1.0, 0.0],
    "N" : [-1.0, 2.0, 0.0, -1.0, -3.0, 1.0, 1.0, -2.0, -1.0, -3.0, -2.0, 5.0, -1.0, -3.0, -1.0, 0.0, -1.0, -3.0, -2.0, -2.0, 0.0],
    "P" : [-1.0, -1.0, -2.0, -3.0, -1.0, 0.0, -2.0, -3.0, -2.0, 1.0, 2.0, -1.0, 5.0, 0.0, -2.0, -1.0, -1.0, -1.0, -1.0, 1.0, 0.0],
    "Q" : [-2.0, -3.0, -3.0, -3.0, -2.0, -3.0, -3.0, -3.0, -1.0, 0.0, 0.0, -3.0, 0.0, 6.0, -4.0, -2.0, -2.0, 1.0, 3.0, -1.0, 0.0],
    "R" : [-1.0, -2.0, -2.0, -1.0, -3.0, -1.0, -1.0, -2.0, -2.0, -3.0, -3.0, -1.0, -2.0, -4.0, 7.0, -1.0, -1.0, -4.0, -3.0, -2.0, 0.0],
    "S" : [1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -2.0, 0.0, -1.0, -2.0, -1.0, 4.0, 1.0, -3.0, -2.0, -2.0, 0.0],
    "T" : [0.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -1.0, -1.0, -1.0, -2.0, -1.0, 1.0, 5.0, -2.0, -2.0, 0.0, 0.0],
    "V" : [-3.0, -3.0, -4.0, -4.0, -2.0, -2.0, -3.0, -2.0, -2.0, -3.0, -2.0, -3.0, -1.0, 1.0, -4.0, -3.0, -2.0, 11.0, 2.0, -3.0, 0.0],
    "W" : [-2.0, -2.0, -2.0, -3.0, -2.0, -1.0, -2.0, -3.0, 2.0, -1.0, -1.0, -2.0, -1.0, 3.0, -3.0, -2.0, -2.0, 2.0, 7.0, -1.0, 0.0],
    "Y" : [0.0, -3.0, -3.0, -3.0, -1.0, -2.0, -2.0, -3.0, -3.0, 3.0, 1.0, -2.0, 1.0, -1.0, -2.0, -2.0, 0.0, -3.0, -1.0, 4.0, 0.0],
    "_" : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}


# modified manually to match the mode
MP3SC_INFO = {
    'A': {'CB': {'bond_lens': 1.5260003, 'bond_angs': 1.9146265, 'bond_dihedral': 2.848366}
    },
    'R': {'CB': {'bond_lens': 1.5260003, 'bond_angs': 1.9146265, 'bond_dihedral': 2.6976738},
     'CG': {'bond_lens': 1.5260003, 'bond_angs': 1.9111352, 'bond_dihedral': -1.2},
     'CD': {'bond_lens': 1.5260003, 'bond_angs': 1.9111352, 'bond_dihedral': -3.141592},
     'NE': {'bond_lens': 1.463, 'bond_angs': 1.9408059, 'bond_dihedral': -3.141592},
     'CZ': {'bond_lens': 1.34, 'bond_angs': 2.1502457, 'bond_dihedral': -3.141592},
     'NH1': {'bond_lens': 1.34, 'bond_angs': 2.094395, 'bond_dihedral': 0.},
     'NH2': {'bond_lens': 1.34, 'bond_angs': 2.094395, 'bond_dihedral': -3.141592}
    },
    'N': {'CB': {'bond_lens': 1.5260003, 'bond_angs': 1.9146265, 'bond_dihedral': 2.8416245},
     'CG': {'bond_lens': 1.5219998, 'bond_angs': 1.9390607, 'bond_dihedral': -1.15},
     'OD1': {'bond_lens': 1.229, 'bond_angs': 2.101376, 'bond_dihedral': -1.}, # spread out w/ mean at -1
     'ND2': {'bond_lens': 1.3349999, 'bond_angs': 2.0350537, 'bond_dihedral': 2.14} # spread out with mean at -4
    },
    'D': {'CB': {'bond_lens': 1.526, 'bond_angs': 1.9146265, 'bond_dihedral': 2.7741134},
     'CG': {'bond_lens': 1.522, 'bond_angs': 1.9390608, 'bond_dihedral': -1.07},
     'OD1': {'bond_lens': 1.25, 'bond_angs': 2.0420356, 'bond_dihedral': -0.2678593},
     'OD2': {'bond_lens': 1.25, 'bond_angs': 2.0420356, 'bond_dihedral': 2.95}
    },
    'C': {'CB': {'bond_lens': 1.5259998, 'bond_angs': 1.9146262, 'bond_dihedral': 2.553627},
     'SG': {'bond_lens': 1.8099997, 'bond_angs': 1.8954275, 'bond_dihedral': -1.07}
    },
    'Q': {'CB': {'bond_lens': 1.5260003, 'bond_angs': 1.9146266, 'bond_dihedral': 2.7262106},
     'CG': {'bond_lens': 1.5260003, 'bond_angs': 1.9111353, 'bond_dihedral': -1.075},
     'CD': {'bond_lens': 1.5219998, 'bond_angs': 1.9390606, 'bond_dihedral': -3.141592},
     'OE1': {'bond_lens': 1.229, 'bond_angs': 2.101376, 'bond_dihedral': -1}, # bimodal at -1, +1 
     'NE2': {'bond_lens': 1.3349998, 'bond_angs': 2.0350537, 'bond_dihedral': 2.14} # bimodal at -2, -4
    },
    'E': {'CB': {'bond_lens': 1.5260003, 'bond_angs': 1.9146267, 'bond_dihedral': 2.7813723},
     'CG': {'bond_lens': 1.5260003, 'bond_angs': 1.9111352, 'bond_dihedral': -1.07}, # bimodal at -1.07, 3.14
     'CD': {'bond_lens': 1.5219998, 'bond_angs': 1.9390606, 'bond_dihedral': -3.0907722155200403},
     'OE1': {'bond_lens': 1.25, 'bond_angs': 2.0420356, 'bond_dihedral': 0.003740118}, # spread out btween -1,1
     'OE2': {'bond_lens': 1.25, 'bond_angs': 2.0420356, 'bond_dihedral': -3.1378527} # spread out btween -4.3, -2.14
    },
    'G': {},
    'H': {'CB': {'bond_lens': 1.5259998, 'bond_angs': 1.9146264, 'bond_dihedral': 2.614421},
     'CG': {'bond_lens': 1.5039998, 'bond_angs': 1.9739674, 'bond_dihedral': -1.05},
     'ND1': {'bond_lens': 1.3850001, 'bond_angs': 2.094395, 'bond_dihedral': -1.41}, # bimodal at -1.4, 1.4
     'CE1': {'bond_lens': 1.3430002, 'bond_angs': 1.8849558, 'bond_dihedral': 3.14},
     'NE2': {'bond_lens': 1.335, 'bond_angs': 1.8849558, 'bond_dihedral': 0.0},
     'CD2': {'bond_lens': 1.3940002, 'bond_angs': 1.8849558, 'bond_dihedral': 0.0}
    },
    'I': {'CB': {'bond_lens': 1.526, 'bond_angs': 1.9146265, 'bond_dihedral': 2.5604365},
     'CG1': {'bond_lens': 1.526, 'bond_angs': 1.9111353, 'bond_dihedral': -1.025},
     'CD1': {'bond_lens': 1.526, 'bond_angs': 1.9111353, 'bond_dihedral': -3.0667439142810267},
     'CG2': {'bond_lens': 1.526, 'bond_angs': 1.9111353, 'bond_dihedral': -3.1225884596454065}
    },
    'L': {'CB': {'bond_lens': 1.5260003, 'bond_angs': 1.9146265, 'bond_dihedral': 2.711971},
     'CG': {'bond_lens': 1.5260003, 'bond_angs': 1.9111352, 'bond_dihedral': -1.15},
     'CD1': {'bond_lens': 1.5260003, 'bond_angs': 1.9111352, 'bond_dihedral': 3.14},
     'CD2': {'bond_lens': 1.5260003, 'bond_angs': 1.9111352, 'bond_dihedral': -1.05}
    },
    'K': {'CB': {'bond_lens': 1.526, 'bond_angs': 1.9146266, 'bond_dihedral': 2.7441595},
     'CG': {'bond_lens': 1.526, 'bond_angs': 1.9111353, 'bond_dihedral': -1.15},
     'CD': {'bond_lens': 1.526, 'bond_angs': 1.9111353, 'bond_dihedral': -3.09},
     'CE': {'bond_lens': 1.526, 'bond_angs': 1.9111353, 'bond_dihedral': 3.092959},
     'NZ': {'bond_lens': 1.4710001, 'bond_angs': 1.940806, 'bond_dihedral': 3.0515378}
    },
    'M': {'CB': {'bond_lens': 1.526, 'bond_angs': 1.9146264, 'bond_dihedral': 2.7051392},
     'CG': {'bond_lens': 1.526, 'bond_angs': 1.9111354, 'bond_dihedral': -1.1},
     'SD': {'bond_lens': 1.8099998, 'bond_angs': 2.001892, 'bond_dihedral': 3.1411812}, # bimodal at 0, 3.14
     'CE': {'bond_lens': 1.8099998, 'bond_angs': 1.7261307, 'bond_dihedral': -0.048235133} # trimodal at -1.41, 0, 1.41
    },
    'F': {'CB': {'bond_lens': 1.5260003, 'bond_angs': 1.9146266, 'bond_dihedral': 2.545154},
     'CG': {'bond_lens': 1.5100001, 'bond_angs': 1.9896755, 'bond_dihedral': -1.2}, # bimodal at -1, 3.14 
     'CD1': {'bond_lens': 1.3999997, 'bond_angs': 2.094395, 'bond_dihedral': 1.41}, # bimodal -1.41, 1.41
     'CE1': {'bond_lens': 1.3999997, 'bond_angs': 2.094395, 'bond_dihedral': 3.141592},
     'CZ': {'bond_lens': 1.3999997, 'bond_angs': 2.094395, 'bond_dihedral': 0.0},
     'CE2': {'bond_lens': 1.3999997, 'bond_angs': 2.094395, 'bond_dihedral': 0.0},
     'CD2': {'bond_lens': 1.3999997, 'bond_angs': 2.094395, 'bond_dihedral': 0.0}
    },
    'P': {'CB': {'bond_lens': 1.5260001, 'bond_angs': 1.9146266, 'bond_dihedral': 3.141592},
     'CG': {'bond_lens': 1.5260001, 'bond_angs': 1.9111352, 'bond_dihedral': -0.707}, # bimodal at -0.7, 0.7
     'CD': {'bond_lens': 1.5260001, 'bond_angs': 1.9111352, 'bond_dihedral': 0.85} # bimodal at -0.85, 0.85
    },
    'S': {'CB': {'bond_lens': 1.5260001, 'bond_angs': 1.9146266, 'bond_dihedral': 2.6017702},
     'OG': {'bond_lens': 1.41, 'bond_angs': 1.9111352, 'bond_dihedral': 1.1}
    },
    'T': {'CB': {'bond_lens': 1.5260001, 'bond_angs': 1.9146265, 'bond_dihedral': 2.55},
     'OG1': {'bond_lens': 1.4099998, 'bond_angs': 1.9111353, 'bond_dihedral': -1.07}, # bimodal at -1 and +1
     'CG2': {'bond_lens': 1.5260001, 'bond_angs': 1.9111353, 'bond_dihedral': -3.05} # bimodal at -1 and -3
    },
    'W': {'CB': {'bond_lens': 1.526, 'bond_angs': 1.9146266, 'bond_dihedral': 3.141592},
     'CG': {'bond_lens': 1.4950002, 'bond_angs': 2.0176008, 'bond_dihedral': -1.2},
     'CD1': {'bond_lens': 1.3520001, 'bond_angs': 2.1816616, 'bond_dihedral': 1.53},
     'NE1': {'bond_lens': 1.3810003, 'bond_angs': 1.8971729, 'bond_dihedral': 3.141592},
     'CE2': {'bond_lens': 1.3799998, 'bond_angs': 1.9477878, 'bond_dihedral': 0.0},
     'CZ2': {'bond_lens': 1.3999999, 'bond_angs': 2.317797, 'bond_dihedral': 3.141592},
     'CH2': {'bond_lens': 1.3999999, 'bond_angs': 2.094395, 'bond_dihedral': 3.141592},
     'CZ3': {'bond_lens': 1.3999999, 'bond_angs': 2.094395, 'bond_dihedral': 0.0},
     'CE3': {'bond_lens': 1.3999999, 'bond_angs': 2.094395, 'bond_dihedral': 0.0},
     'CD2': {'bond_lens': 1.404, 'bond_angs': 2.094395, 'bond_dihedral': 0.0}
    },
    'Y': {'CB': {'bond_lens': 1.5260003, 'bond_angs': 1.9146266, 'bond_dihedral': 3.1},
     'CG': {'bond_lens': 1.5100001, 'bond_angs': 1.9896754, 'bond_dihedral': -1.1},
     'CD1': {'bond_lens': 1.3999997, 'bond_angs': 2.094395, 'bond_dihedral': 1.36},
     'CE1': {'bond_lens': 1.3999997, 'bond_angs': 2.094395, 'bond_dihedral': 3.141592},
     'CZ': {'bond_lens': 1.4090003, 'bond_angs': 2.094395, 'bond_dihedral': 0.0},
     'OH': {'bond_lens': 1.3640002, 'bond_angs': 2.094395, 'bond_dihedral': 3.141592},
     'CE2': {'bond_lens': 1.4090003, 'bond_angs': 2.094395, 'bond_dihedral': 0.0},
     'CD2': {'bond_lens': 1.3999997, 'bond_angs': 2.094395, 'bond_dihedral': 0.0}
    },
    'V': {'CB': {'bond_lens': 1.5260003, 'bond_angs': 1.9146266, 'bond_dihedral': 2.55},
     'CG1': {'bond_lens': 1.5260003, 'bond_angs': 1.9111352, 'bond_dihedral': 3.141592},
     'CG2': {'bond_lens': 1.5260003, 'bond_angs': 1.9111352, 'bond_dihedral': -1.1}
    },

    '_': {}
}

# experimentally checked distances
FF = {"MIN_DISTS": {1: 1.180, # shortest =N or =O bond
                    2: 2.138, # N-N in histidine group
                    3: 2.380}, # N-N in backbone (N-CA-C-N) 
      "MAX_DISTS": {i: 1.840*i for i in range(1, 5+1)} # 1.84 is longest -S bond found,
     } 

ATOM_TOKEN_IDS = set(["", "N", "CA", "C", "O"])
ATOM_TOKEN_IDS = {k: i for i,k in enumerate(sorted( 
                    ATOM_TOKEN_IDS.union( set(
                        [name for k,v in SC_BUILD_INFO.items() for name in v["atom-names"]]
                    ) )
                ))}

#################
##### DOERS #####
#################

def make_cloud_mask(aa):
    """ relevent points will be 1. paddings will be 0. """
    mask = np.zeros(14)
    if aa != "_":
        n_atoms = 4+len( SC_BUILD_INFO[aa]["atom-names"] )
        mask[:n_atoms] = True
    return mask

def make_bond_mask(aa):
    """ Gives the length of the bond originating each atom. """
    mask = np.zeros(14)
    # backbone
    if aa != "_":
        mask[0] = BB_BUILD_INFO["BONDLENS"]['c-n']
        mask[1] = BB_BUILD_INFO["BONDLENS"]['n-ca']
        mask[2] = BB_BUILD_INFO["BONDLENS"]['ca-c']
        mask[3] = BB_BUILD_INFO["BONDLENS"]['c-o']
        # sidechain - except padding token 
        if aa in SC_BUILD_INFO.keys():
            for i,bond in enumerate(SC_BUILD_INFO[aa]['bonds-vals']):
                mask[4+i] = bond
    return mask

def make_theta_mask(aa):
    """ Gives the theta of the bond originating each atom. """
    mask = np.zeros(14)
    # backbone
    if aa != "_":
        mask[0] = BB_BUILD_INFO["BONDANGS"]['ca-c-n'] # nitrogen
        mask[1] = BB_BUILD_INFO["BONDANGS"]['c-n-ca'] # c_alpha
        mask[2] = BB_BUILD_INFO["BONDANGS"]['n-ca-c'] # carbon
        mask[3] = BB_BUILD_INFO["BONDANGS"]['ca-c-o'] # oxygen
        # sidechain
        for i,theta in enumerate(SC_BUILD_INFO[aa]['angles-vals']):
            mask[4+i] = theta
    return mask

def make_torsion_mask(aa, fill=False):
    """ Gives the dihedral of the bond originating each atom. """
    mask = np.zeros(14)
    if aa != "_":
        # backbone
        mask[0] = BB_BUILD_INFO["BONDTORSIONS"]['n-ca-c-n'] # psi
        mask[1] = BB_BUILD_INFO["BONDTORSIONS"]['ca-n-c-ca'] # omega
        mask[2] = BB_BUILD_INFO["BONDTORSIONS"]['c-n-ca-c'] # psi
        mask[3] = BB_BUILD_INFO["BONDTORSIONS"]['n-ca-c-o'] # oxygen
        # sidechain
        for i, torsion in enumerate(SC_BUILD_INFO[aa]['torsion-vals']):
            if fill: 
                mask[4+i] = MP3SC_INFO[aa][ SC_BUILD_INFO[aa]["atom-names"][i] ]["bond_dihedral"]
            else: 
                # https://github.com/jonathanking/sidechainnet/blob/master/sidechainnet/structure/StructureBuilder.py#L372
                # 999 is an anotation -- change later || same for 555
                mask[4+i] = np.nan if torsion == 'p' else 999 if torsion == "i" else torsion
    return mask

def make_idx_mask(aa):
    """ Gives the idxs of the 3 previous points. """
    mask = np.zeros((11, 3))
    if aa != "_":
        # backbone
        mask[0, :] = np.arange(3) 
        # sidechain
        mapper = {"N": 0, "CA": 1, "C":2,  "CB": 4}
        for i, torsion in enumerate(SC_BUILD_INFO[aa]['torsion-names']):
            # get all the atoms forming the dihedral
            torsions = [x.rstrip(" ") for x in torsion.split("-")]
            # for each atom
            for n, torsion in enumerate(torsions[:-1]):
                # get the index of the atom in the coords array
                loc = mapper[torsion] if torsion in mapper.keys() else 4 + SC_BUILD_INFO[aa]['atom-names'].index(torsion)
                # set position to index
                mask[i+1][n] = loc
    return mask

def make_atom_token_mask(aa):
    """ Return the tokens for each atom in the aa. """
    mask = np.zeros(14)
    # get atom id
    if aa != "_":
        atom_list = ["N", "CA", "C", "O"] + SC_BUILD_INFO[ aa ]["atom-names"]
        for i,atom in enumerate(atom_list):
            mask[i] = ATOM_TOKEN_IDS[atom]
    return mask


###################
##### GETTERS #####
###################
INDEX2AAS = "ACDEFGHIKLMNPQRSTVWY_"
AAS2INDEX = {aa:i for i,aa in enumerate(INDEX2AAS)}
SUPREME_INFO = {k: {"cloud_mask": make_cloud_mask(k),
                    "bond_mask": make_bond_mask(k),
                    "theta_mask": make_theta_mask(k),
                    "torsion_mask": make_torsion_mask(k),
                    "torsion_mask_filled": make_torsion_mask(k, fill=True),
                    "idx_mask": make_idx_mask(k),
                    "atom_token_mask": make_atom_token_mask(k),
                    "rigid_idx_mask": SC_BUILD_INFO[k]['rigid-frames-idxs'],
                    } 
                for k in INDEX2AAS}

