"""SYNTHETIC document packet generator (Phase 2): loan packets and insurance submissions with ground truth.

Everything this package makes is synthetic: fictional institutions, made-up people and businesses, and documents
that print "SYNTHETIC - for demonstration only" on every page. The extraction pipeline must never import this
package (only scripts/make_packets.py, scripts/eval_*.py and tests may); a test enforces it.
"""

from uwcopilot.docintel.synthetic.generate import generate_packets
from uwcopilot.docintel.synthetic.packet import build_packet

__all__ = ["build_packet", "generate_packets"]
