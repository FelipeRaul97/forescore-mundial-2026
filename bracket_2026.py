"""
bracket_2026.py — Estructura oficial del bracket del Mundial 2026.

R32 (matches 73-88): cada slot definido por (tipo, grupo) donde
  tipo in {W: winner, R: runner-up, T: best-third de un cluster}
R16+ : el árbol fijo que conecta los ganadores.

Fuente: FIFA / USA TODAY draw 5-dic-2025.
"""

# Cada match R32: (lado_A, lado_B)
# 'W:E' = winner grupo E ; 'R:A' = runner-up grupo A ; 'T:ABCDF' = best third entre esos grupos
R32 = {
    73: ("R:A", "R:B"),
    74: ("W:E", "T:ABCDF"),
    75: ("W:F", "R:C"),
    76: ("W:C", "R:F"),
    77: ("W:I", "T:CDFGH"),
    78: ("R:E", "R:I"),
    79: ("W:A", "T:CEFHI"),
    80: ("W:L", "T:EHIJK"),
    81: ("W:D", "T:BEFIJ"),
    82: ("W:G", "T:AEHIJ"),
    83: ("R:K", "R:L"),
    84: ("W:H", "R:J"),
    85: ("W:B", "T:EFGIJ"),
    86: ("W:J", "R:H"),
    87: ("W:K", "T:DEIJL"),
    88: ("R:D", "R:G"),
}

# R16: cada match recibe ganadores de dos matches previos
R16 = {
    89: (74, 77),
    90: (73, 75),
    91: (76, 78),
    92: (79, 80),
    93: (83, 84),
    94: (81, 82),
    95: (86, 88),
    96: (85, 87),
}

QF = {
    97: (89, 90),
    98: (93, 94),
    99: (91, 92),
    100: (95, 96),
}

SF = {
    101: (97, 98),
    102: (99, 100),
}

FINAL = (101, 102)
