use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

pub mod attacks;
pub mod board;
pub mod evaluation;
pub mod game;
pub mod moves;
pub mod nnue;
pub mod search;
pub mod simd;
pub mod tiles;
mod utils;

// Initialize panic hook for better error messages in WASM
// This will show actual line numbers instead of just "unreachable"
#[cfg(feature = "debug")]
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

use crate::moves::{SpatialIndices, set_world_bounds};
use board::{Board, Coordinate, Piece, PieceType, PlayerColor};
use evaluation::calculate_initial_material;
use game::{EnPassantState, GameState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Classical,
    ConfinedClassical,
    ClassicalPlus,
    CoaIP,
    CoaIPHO,
    CoaIPRO,
    CoaIPNO,
    Palace,
    Pawndard,
    Core,
    Standarch,
    SpaceClassic,
    Space,
    Abundance,
    PawnHorde,
    Knightline,
    Obstocean,
    Chess,
}

impl Variant {
    #[cfg(any(test, not(target_arch = "wasm32")))]
    pub fn starting_icn(&self) -> &'static str {
        match self {
            Variant::Classical => {
                "w 0/100 1 (8|1) P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|R1,1+|R8,1+|r1,8+|r8,8+|N2,1|N7,1|n2,8|n7,8|B3,1|B6,1|b3,8|b6,8|Q4,1|q4,8|K5,1+|k5,8+"
            }
            Variant::ConfinedClassical => {
                "w 0/100 1 (8|1) -1000000000000000,1000000000000009,-1000000000000000,1000000000000009 P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|R1,1+|R8,1+|r1,8+|r8,8+|N2,1|N7,1|n2,8|n7,8|B3,1|B6,1|b3,8|b6,8|Q4,1|q4,8|K5,1+|k5,8+|ob0,0|ob0,1|ob0,2|ob0,7|ob0,8|ob0,9|ob9,0|ob9,1|ob9,2|ob9,7|ob9,8|ob9,9|ob1,0|ob2,0|ob3,0|ob4,0|ob5,0|ob6,0|ob7,0|ob8,0|ob1,9|ob2,9|ob3,9|ob4,9|ob5,9|ob6,9|ob7,9|ob8,9"
            }
            Variant::ClassicalPlus => {
                "w 0/100 1 (8|1) p1,9+|p2,9+|p3,9+|p6,9+|p7,9+|p8,9+|p0,8+|r1,8+|n2,8|b3,8|q4,8|k5,8+|b6,8|n7,8|r8,8+|p9,8+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|p3,5+|p6,5+|P3,4+|P6,4+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P0,1+|R1,1+|N2,1|B3,1|Q4,1|K5,1+|B6,1|N7,1|R8,1+|P9,1+|P1,0+|P2,0+|P3,0+|P6,0+|P7,0+|P8,0+"
            }
            Variant::CoaIP => {
                "w 0/100 1 (8;n,b,r,q,gu,ch,ha|1;n,b,r,q,gu,ch,ha) P-2,1+|P-1,2+|P0,2+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P9,2+|P10,2+|P11,1+|P-4,-6+|P-3,-5+|P-2,-4+|P-1,-5+|P0,-6+|P9,-6+|P10,-5+|P11,-4+|P12,-5+|P13,-6+|p-2,8+|p-1,7+|p0,7+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|p9,7+|p10,7+|p11,8+|p-4,15+|p-3,14+|p-2,13+|p-1,14+|p0,15+|p9,15+|p10,14+|p11,13+|p12,14+|p13,15+|HA-2,-6|HA11,-6|ha-2,15|ha11,15|R-1,1|R10,1|r-1,8|r10,8|CH0,1|CH9,1|ch0,8|ch9,8|GU1,1+|GU8,1+|gu1,8+|gu8,8+|N2,1|N7,1|n2,8|n7,8|B3,1|B6,1|b3,8|b6,8|Q4,1|q4,8|K5,1+|k5,8+"
            }
            Variant::CoaIPHO => {
                "w 0/100 1 (8;n,b,r,q,gu,ch,ha,hu|1;n,b,r,q,gu,ch,ha,hu) p-4,14+|ha-2,14|p0,14+|p9,14+|ha11,14|p13,14+|p-3,13+|p-1,13+|p10,13+|p12,13+|p-2,12+|p11,12+|gu-1,9|hu0,9|ch1,9|ch8,9|hu9,9|gu10,9|p-1,8+|p0,8+|r1,8+|n2,8|b3,8|q4,8|k5,8+|b6,8|n7,8|r8,8+|p9,8+|p10,8+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P-1,1+|P0,1+|R1,1+|N2,1|B3,1|Q4,1|K5,1+|B6,1|N7,1|R8,1+|P9,1+|P10,1+|GU-1,0|HU0,0|CH1,0|CH8,0|HU9,0|GU10,0|P-2,-3+|P11,-3+|P-3,-4+|P-1,-4+|P10,-4+|P12,-4+|P-4,-5+|HA-2,-5|P0,-5+|P9,-5+|HA11,-5|P13,-5+"
            }
            Variant::CoaIPRO => {
                "w 0/100 1 (8;n,b,r,q,gu,ch,ro|1;n,b,r,q,gu,ch,ro) P-2,1+|P-1,2+|P0,2+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P9,2+|P10,2+|P11,1+|P-4,-6+|P-3,-5+|P-2,-4+|P-1,-5+|P0,-6+|P9,-6+|P10,-5+|P11,-4+|P12,-5+|P13,-6+|p-2,8+|p-1,7+|p0,7+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|p9,7+|p10,7+|p11,8+|p-4,15+|p-3,14+|p-2,13+|p-1,14+|p0,15+|p9,15+|p10,14+|p11,13+|p12,14+|p13,15+|R-1,1|R10,1|r-1,8|r10,8|CH0,1|CH9,1|ch0,8|ch9,8|GU1,1+|GU8,1+|gu1,8+|gu8,8+|N2,1|N7,1|n2,8|n7,8|B3,1|B6,1|b3,8|b6,8|Q4,1|q4,8|K5,1+|k5,8+|RO-2,-6|RO11,-6|ro-2,15|ro11,15"
            }
            Variant::CoaIPNO => {
                "w 0/100 1 (8;n,b,r,q,gu,ch,nr|1;n,b,r,q,gu,ch,nr) P-2,1+|P-1,2+|P0,2+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P9,2+|P10,2+|P11,1+|P-4,-6+|P-3,-5+|P-2,-4+|P-1,-5+|P0,-6+|P9,-6+|P10,-5+|P11,-4+|P12,-5+|P13,-6+|p-2,8+|p-1,7+|p0,7+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|p9,7+|p10,7+|p11,8+|p-4,15+|p-3,14+|p-2,13+|p-1,14+|p0,15+|p9,15+|p10,14+|p11,13+|p12,14+|p13,15+|R-1,1|R10,1|r-1,8|r10,8|CH0,1|CH9,1|ch0,8|ch9,8|GU1,1+|GU8,1+|gu1,8+|gu8,8+|N2,1|N7,1|n2,8|n7,8|B3,1|B6,1|b3,8|b6,8|Q4,1|q4,8|K5,1+|k5,8+|nr-2,16|nr11,16|NR-2,-7|NR11,-7"
            }
            Variant::Palace => {
                "w 0/100 1 (4;n,b,r,q,am|2;n,b,r,q,am) K4,1|Q5,1|P6,2+|P5,2+|P4,2+|P3,2+|P2,2+|P1,2+|p1,4+|p2,4+|p3,4+|p4,4+|p5,4+|p6,4+|N6,1|AM3,1|Q2,1|N1,1|n1,5|n6,5|k4,5|q5,5|q2,5|am3,5|P6,-1+|P7,-1+|P8,-1+|P9,-1+|P1,-1+|P0,-1+|P-1,-1+|P-2,-1+|P2,-2+|P-3,-2+|P5,-2+|P10,-2+|p7,7+|p6,7+|p8,7+|p9,7+|p1,7+|p0,7+|p-1,7+|p-2,7+|p-3,8+|p2,8+|p5,8+|p10,8+|r-1,8|r-2,8|r8,8|r9,8|R8,-2|R9,-2|R-1,-2|R-2,-2|B0,-2|B1,-2|B7,-2|B6,-2|b0,8|b1,8|b7,8|b6,8"
            }
            Variant::Pawndard => {
                "w 0/100 1 (8|1) b4,14|b5,14|r4,12|r5,12|p2,10+|p3,10+|p6,10+|p7,10+|p1,9+|p8,9+|p0,8+|n2,8|n3,8|k4,8+|q5,8|n6,8|n7,8|p9,8+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|P1,5+|p2,5+|P3,5+|p6,5+|P7,5+|p8,5+|p1,4+|P2,4+|p3,4+|P6,4+|p7,4+|P8,4+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P0,1+|N2,1|N3,1|Q4,1|K5,1+|N6,1|N7,1|P9,1+|P1,0+|P8,0+|P2,-1+|P3,-1+|P6,-1+|P7,-1+|R4,-3|R5,-3|B4,-5|B5,-5"
            }
            Variant::Core => {
                "w 0/100 1 (8|1) p-1,10+|p3,10+|p4,10+|p5,10+|p6,10+|p10,10+|p0,9+|p9,9+|n0,8|r1,8+|n2,8|b3,8|q4,8|k5,8+|b6,8|n7,8|r8,8+|n9,8|p-2,7+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|p11,7+|p-3,6+|p12,6+|p1,5+|P2,5+|P7,5+|p8,5+|P1,4+|p2,4+|p7,4+|P8,4+|P-3,3+|P12,3+|P-2,2+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P11,2+|N0,1|R1,1+|N2,1|B3,1|Q4,1|K5,1+|B6,1|N7,1|R8,1+|N9,1|P0,0+|P9,0+|P-1,-1+|P3,-1+|P4,-1+|P5,-1+|P6,-1+|P10,-1+"
            }
            Variant::Standarch => {
                "w 0/100 1 (8;n,b,r,q,ch,ar|1;n,b,r,q,ch,ar) p4,11+|p5,11+|p1,10+|p2,10+|p3,10+|p6,10+|p7,10+|p8,10+|p0,9+|ar4,9|ch5,9|p9,9+|p0,8+|r1,8+|n2,8|b3,8|q4,8|k5,8+|b6,8|n7,8|r8,8+|p9,8+|p0,7+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|p9,7+|P0,2+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P9,2+|P0,1+|R1,1+|N2,1|B3,1|Q4,1|K5,1+|B6,1|N7,1|R8,1+|P9,1+|P0,0+|AR4,0|CH5,0|P9,0+|P1,-1+|P2,-1+|P3,-1+|P6,-1+|P7,-1+|P8,-1+|P4,-2+|P5,-2+"
            }
            Variant::SpaceClassic => {
                "w 0/100 1 (8|1) p-3,18+|r2,18|b4,18|b5,18|r7,18|p12,18+|p-4,17+|p13,17+|p-5,16+|p14,16+|p3,9+|p4,9+|p5,9+|p6,9+|n3,8|k4,8|q5,8|n6,8|p-6,7+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|p-8,6+|p-7,6+|p16,6+|p17,6+|p-9,5+|p18,5+|P-9,4+|P18,4+|P-8,3+|P-7,3+|P16,3+|P17,3+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P15,2+|N3,1|K4,1|Q5,1|N6,1|P3,0+|P4,0+|P5,0+|P6,0+|P-5,-7+|P14,-7+|P-4,-8+|P13,-8+|P-3,-9+|R2,-9|B4,-9|B5,-9|R7,-9|P12,-9+"
            }
            Variant::Space => {
                "w 0/100 1 (4;n,b,r,q,ha,ce,ar,ch|-3;n,b,r,q,ha,ce,ar,ch) q4,31|ch4,23|p-12,18+|b4,18|p20,18+|p-11,17+|ar-10,17|p0,17+|b4,17|p8,17+|ar18,17|p19,17+|p-11,16+|p-10,16+|p-1,16+|p9,16+|p18,16+|p19,16+|p-1,15+|r0,15|ha4,15|r8,15|p9,15+|p3,6+|p4,6+|p5,6+|p2,5+|k4,5|p6,5+|n1,4|ce4,4|n7,4|p-10,3+|p-1,3+|p0,3+|p2,3+|p3,3+|p4,3+|p5,3+|p6,3+|p8,3+|p9,3+|p-12,2+|p-11,2+|p19,2+|p20,2+|p-13,1+|p21,1+|P-13,0+|P21,0+|P-12,-1+|P-11,-1+|P19,-1+|P20,-1+|P-1,-2+|P0,-2+|P2,-2+|P3,-2+|P4,-2+|P5,-2+|P6,-2+|P8,-2+|P9,-2+|P18,-2+|N1,-3|CE4,-3|N7,-3|P2,-4+|K4,-4|P6,-4+|P3,-5+|P4,-5+|P5,-5+|P-1,-14+|R0,-14|HA4,-14|R8,-14|P9,-14+|P-11,-15+|P-10,-15+|P-1,-15+|P9,-15+|P18,-15+|P19,-15+|P-11,-16+|AR-10,-16|P0,-16+|B4,-16|P8,-16+|AR18,-16|P19,-16+|P-12,-17+|B4,-17|P20,-17+|CH4,-22|Q4,-30"
            }
            Variant::Abundance => {
                "w 0/100 1 (6;n,b,r,q,gu,ha,ch|-6;n,b,r,q,gu,ha,ch) p-3,10+|ha-2,10|ha-1,10|r0,10|ha1,10|ha2,10|p3,10+|p-2,9+|p-1,9+|p1,9+|p2,9+|p-5,6+|gu-4,6|r-3,6+|b-2,6|b-1,6|k0,6+|b1,6|b2,6|r3,6+|gu4,6|p5,6+|p-4,5+|gu-3,5|n-1,5|q0,5|n1,5|gu3,5|p4,5+|p-3,4+|p-2,4+|gu-1,4|ch0,4|gu1,4|p2,4+|p3,4+|p-1,3+|p0,3+|p1,3+|P-1,-3+|P0,-3+|P1,-3+|P-3,-4+|P-2,-4+|GU-1,-4|CH0,-4|GU1,-4|P2,-4+|P3,-4+|P-4,-5+|GU-3,-5|N-1,-5|Q0,-5|N1,-5|GU3,-5|P4,-5+|P-5,-6+|GU-4,-6|R-3,-6+|B-2,-6|B-1,-6|K0,-6+|B1,-6|B2,-6|R3,-6+|GU4,-6|P5,-6+|P-2,-9+|P-1,-9+|P1,-9+|P2,-9+|P-3,-10+|HA-2,-10|HA-1,-10|R0,-10|HA1,-10|HA2,-10|P3,-10+"
            }
            Variant::PawnHorde => {
                "w 0/100 1 (2|-7) checkmate,allpiecescaptured k5,2+|q4,2|r1,2+|n7,2|n2,2|r8,2+|b3,2|b6,2|P2,-1+|P3,-1+|P6,-1+|P7,-1+|P1,-2+|P2,-2+|P4,-2+|P5,-2+|P6,-2+|P7,-2+|P8,-2+|P1,-3+|P2,-3+|P4,-3+|P5,-3+|P6,-3+|P7,-3+|P8,-3+|P1,-4+|P2,-4+|P4,-4+|P5,-4+|P6,-4+|P7,-4+|P8,-4+|P1,-5+|P2,-5+|P4,-5+|P5,-5+|P6,-5+|P7,-5+|P8,-5+|P1,-6+|P2,-6+|P4,-6+|P5,-6+|P6,-6+|P7,-6+|P8,-6+|P3,-2+|P3,-3+|P3,-4+|P3,-5+|P3,-6+|P1,-7+|P2,-7+|P3,-7+|P4,-7+|P5,-7+|P6,-7+|P7,-7+|P8,-7+|P0,-6+|P0,-7+|P9,-6+|P9,-7+|p9,2+|p1,1+|p2,1+|p3,1+|p4,1+|p5,1+|p6,1+|p7,1+|p8,1+|p0,2+"
            }
            Variant::Knightline => {
                "w 0/100 1 (8;n,q|1;n,q) k5,8|n3,8|n4,8|n6,8|n7,8|p-5,7+|p-4,7+|p-3,7+|p-2,7+|p-1,7+|p0,7+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|p9,7+|p10,7+|p11,7+|p12,7+|p13,7+|p14,7+|p15,7+|K5,1|N3,1|N4,1|N6,1|N7,1|P-5,2+|P-4,2+|P-3,2+|P-2,2+|P-1,2+|P0,2+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P9,2+|P10,2+|P11,2+|P12,2+|P13,2+|P14,2+|P15,2+"
            }
            Variant::Obstocean => {
                "w 0/100 1 (8|1) -6,15,-3,12 ob-6,12|ob-5,12|ob-4,12|ob-3,12|ob-2,12|ob-1,12|ob0,12|ob1,12|ob2,12|ob3,12|ob4,12|ob5,12|ob6,12|ob7,12|ob8,12|ob9,12|ob10,12|ob11,12|ob12,12|ob13,12|ob14,12|ob15,12|ob-6,11|ob-5,11|ob-4,11|ob-3,11|ob-2,11|ob-1,11|ob0,11|ob1,11|ob2,11|ob3,11|ob4,11|ob5,11|ob6,11|ob7,11|ob8,11|ob9,11|ob10,11|ob11,11|ob12,11|ob13,11|ob14,11|ob15,11|ob-6,10|ob-5,10|ob-4,10|ob-3,10|ob-2,10|ob-1,10|ob0,10|ob1,10|ob2,10|ob3,10|ob4,10|ob5,10|ob6,10|ob7,10|ob8,10|ob9,10|ob10,10|ob11,10|ob12,10|ob13,10|ob14,10|ob15,10|ob-6,9|ob-5,9|ob-4,9|ob-3,9|ob-2,9|ob-1,9|ob0,9|ob1,9|ob2,9|ob3,9|ob4,9|ob5,9|ob6,9|ob7,9|ob8,9|ob9,9|ob10,9|ob11,9|ob12,9|ob13,9|ob14,9|ob15,9|ob-6,8|ob-5,8|ob-4,8|ob-3,8|ob-2,8|ob-1,8|ob0,8|r1,8+|n2,8|b3,8|q4,8|k5,8+|b6,8|n7,8|r8,8+|ob9,8|ob10,8|ob11,8|ob12,8|ob13,8|ob14,8|ob15,8|ob-6,7|ob-5,7|ob-4,7|ob-3,7|ob-2,7|ob-1,7|ob0,7|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|ob9,7|ob10,7|ob11,7|ob12,7|ob13,7|ob14,7|ob15,7|ob-6,6|ob-5,6|ob-4,6|ob-3,6|ob-2,6|ob-1,6|ob0,6|ob1,6|ob2,6|ob3,6|ob4,6|ob5,6|ob6,6|ob7,6|ob8,6|ob9,6|ob10,6|ob11,6|ob12,6|ob13,6|ob14,6|ob15,6|ob-6,5|ob-5,5|ob-4,5|ob-3,5|ob-2,5|ob-1,5|ob0,5|ob1,5|ob2,5|ob3,5|ob4,5|ob5,5|ob6,5|ob7,5|ob8,5|ob9,5|ob10,5|ob11,5|ob12,5|ob13,5|ob14,5|ob15,5|ob-6,4|ob-5,4|ob-4,4|ob-3,4|ob-2,4|ob-1,4|ob0,4|ob1,4|ob2,4|ob3,4|ob4,4|ob5,4|ob6,4|ob7,4|ob8,4|ob9,4|ob10,4|ob11,4|ob12,4|ob13,4|ob14,4|ob15,4|ob-6,3|ob-5,3|ob-4,3|ob-3,3|ob-2,3|ob-1,3|ob0,3|ob1,3|ob2,3|ob3,3|ob4,3|ob5,3|ob6,3|ob7,3|ob8,3|ob9,3|ob10,3|ob11,3|ob12,3|ob13,3|ob14,3|ob15,3|ob-6,2|ob-5,2|ob-4,2|ob-3,2|ob-2,2|ob-1,2|ob0,2|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|ob9,2|ob10,2|ob11,2|ob12,2|ob13,2|ob14,2|ob15,2|ob-6,1|ob-5,1|ob-4,1|ob-3,1|ob-2,1|ob-1,1|ob0,1|R1,1+|N2,1|B3,1|Q4,1|K5,1+|B6,1|N7,1|R8,1+|ob9,1|ob10,1|ob11,1|ob12,1|ob13,1|ob14,1|ob15,1|ob-6,0|ob-5,0|ob-4,0|ob-3,0|ob-2,0|ob-1,0|ob0,0|ob1,0|ob2,0|ob3,0|ob4,0|ob5,0|ob6,0|ob7,0|ob8,0|ob9,0|ob10,0|ob11,0|ob12,0|ob13,0|ob14,0|ob15,0|ob-6,-1|ob-5,-1|ob-4,-1|ob-3,-1|ob-2,-1|ob-1,-1|ob0,-1|ob1,-1|ob2,-1|ob3,-1|ob4,-1|ob5,-1|ob6,-1|ob7,-1|ob8,-1|ob9,-1|ob10,-1|ob11,-1|ob12,-1|ob13,-1|ob14,-1|ob15,-1|ob-6,-2|ob-5,-2|ob-4,-2|ob-3,-2|ob-2,-2|ob-1,-2|ob0,-2|ob1,-2|ob2,-2|ob3,-2|ob4,-2|ob5,-2|ob6,-2|ob7,-2|ob8,-2|ob9,-2|ob10,-2|ob11,-2|ob12,-2|ob13,-2|ob14,-2|ob15,-2|ob-6,-3|ob-5,-3|ob-4,-3|ob-3,-3|ob-2,-3|ob-1,-3|ob0,-3|ob1,-3|ob2,-3|ob3,-3|ob4,-3|ob5,-3|ob6,-3|ob7,-3|ob8,-3|ob9,-3|ob10,-3|ob11,-3|ob12,-3|ob13,-3|ob14,-3|ob15,-3"
            }
            Variant::Chess => {
                "w 0/100 1 (8|1) 1,8,1,8 P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|R1,1+|R8,1+|r1,8+|r8,8+|N2,1|N7,1|n2,8|n7,8|B3,1|B6,1|b3,8|b6,8|Q4,1|q4,8|K5,1+|k5,8+"
            }
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "Classical" => Variant::Classical,
            "Confined_Classical" => Variant::ConfinedClassical,
            "Classical_Plus" => Variant::ClassicalPlus,
            "CoaIP" => Variant::CoaIP,
            "CoaIP_HO" => Variant::CoaIPHO,
            "CoaIP_RO" => Variant::CoaIPRO,
            "CoaIP_NO" => Variant::CoaIPNO,
            "Palace" => Variant::Palace,
            "Pawndard" => Variant::Pawndard,
            "Core" => Variant::Core,
            "Standarch" => Variant::Standarch,
            "Space_Classic" => Variant::SpaceClassic,
            "Space" => Variant::Space,
            "Abundance" => Variant::Abundance,
            "Pawn_Horde" => Variant::PawnHorde,
            "Knightline" => Variant::Knightline,
            "Obstocean" => Variant::Obstocean,
            "Chess" => Variant::Chess,
            _ => Variant::Classical, // Default fallback
        }
    }

    pub fn get_default_bounds(&self) -> (i64, i64, i64, i64) {
        match self {
            Variant::Chess => (1, 8, 1, 8),
            Variant::Obstocean => (-6, 15, -3, 12),
            _ => (
                -1_000_000_000_000_000,
                1_000_000_000_000_000,
                -1_000_000_000_000_000,
                1_000_000_000_000_000,
            ),
        }
    }
}

impl std::str::FromStr for Variant {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Variant::parse(s))
    }
}

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    pub fn group(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    pub fn groupEnd();
}

#[wasm_bindgen]
pub fn reset_engine_state() {
    crate::search::reset_search_state();
}

// Lazy SMP via wasm-bindgen-rayon
// When the multithreading feature is enabled, we re-export init_thread_pool which
// allows JS to initialize a rayon thread pool with shared WebAssembly memory.
// The thread count is controlled by JS (defaults to 1 for single-threaded mode).
#[cfg(all(target_arch = "wasm32", feature = "multithreading"))]
pub use wasm_bindgen_rayon::init_thread_pool;

#[derive(Serialize, Deserialize)]
pub struct JsMove {
    pub from: String, // "x,y"
    pub to: String,   // "x,y"
    pub promotion: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct JsMoveWithEval {
    pub from: String, // "x,y"
    pub to: String,   // "x,y"
    pub promotion: Option<String>,
    pub eval: i32,    // centipawn score from side-to-move's perspective
    pub depth: usize, // depth reached
}

/// A single PV line for MultiPV output
#[derive(Serialize, Deserialize)]
pub struct JsPVLine {
    pub from: String, // "x,y"
    pub to: String,   // "x,y"
    pub promotion: Option<String>,
    pub eval: i32,       // centipawn score from side-to-move's perspective
    pub depth: usize,    // depth searched
    pub pv: Vec<String>, // full PV as array of "x,y->x,y" strings
}

#[derive(Deserialize)]
struct JsFullGame {
    board: JsBoard,
    turn: String,
    /// All special rights - includes castling (kings/rooks) AND pawn double-move rights
    #[serde(default)]
    special_rights: Vec<String>,
    en_passant: Option<JsEnPassant>,
    halfmove_clock: u32,
    fullmove_number: u32,
    #[serde(default)]
    move_history: Vec<JsMoveHistory>,
    #[serde(default)]
    game_rules: Option<JsGameRules>,
    #[serde(default)]
    world_bounds: Option<JsWorldBounds>,
    #[serde(default)]
    clock: Option<JsClock>,
    #[serde(default)]
    variant: Option<String>,
    /// Optional strength hint from the UI/JS side (1=Easy, 2=Medium, 3=Hard).
    #[serde(default)]
    strength_level: Option<u32>,
}

#[derive(Deserialize, Default)]
struct JsGameRules {
    #[serde(default)]
    promotion_ranks: Option<JsPromotionRanks>,
    #[serde(default)]
    promotions_allowed: Option<Vec<String>>,
    #[serde(default)]
    move_rule: Option<u32>,
    #[serde(default)]
    win_conditions: Option<JsWinConditions>,
}

/// Win conditions per side, as received from JavaScript.
/// Each side has an array of conditions; we only use the first.
#[derive(Deserialize, Default)]
struct JsWinConditions {
    #[serde(default)]
    white: Vec<String>,
    #[serde(default)]
    black: Vec<String>,
}

#[derive(Deserialize)]
struct JsPromotionRanks {
    white: Vec<String>, // String because BigInt serializes as string
    black: Vec<String>,
}

#[derive(Deserialize)]
struct JsWorldBounds {
    left: String,
    right: String,
    bottom: String,
    top: String,
}

#[derive(Deserialize, Clone, Copy)]
struct JsClock {
    /// Remaining time for White in milliseconds
    wtime: u64,
    /// Remaining time for Black in milliseconds
    btime: u64,
    /// Increment for White in milliseconds
    winc: u64,
    /// Increment for Black in milliseconds
    binc: u64,
}

#[derive(Deserialize)]
struct JsMoveHistory {
    from: String, // "x,y"
    to: String,   // "x,y"
    #[serde(default)]
    promotion: Option<String>,
}

#[derive(Deserialize)]
struct JsBoard {
    pieces: Vec<JsPiece>,
}

#[derive(Deserialize)]
struct JsPiece {
    x: String,
    y: String,
    piece_type: String,
    player: String,
}

#[derive(Deserialize)]
struct JsEnPassant {
    square: String,      // "x,y"
    pawn_square: String, // "x,y"
}

#[cfg(feature = "eval_tuning")]
#[derive(Serialize)]
struct JsEvalWithFeatures {
    eval: i32,
    features: crate::evaluation::EvalFeatures,
}

#[cfg(target_arch = "wasm32")]
fn get_random_seed() -> u64 {
    (js_sys::Math::random() * (u32::MAX as f64)) as u64
        ^ ((js_sys::Math::random() * (u32::MAX as f64)) as u64).rotate_left(32)
}

#[cfg(not(target_arch = "wasm32"))]
fn get_random_seed() -> u64 {
    use std::time::SystemTime;
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    duration.as_nanos() as u64
}

#[wasm_bindgen]
pub struct Engine {
    game: GameState,
    clock: Option<JsClock>,
    strength_level: Option<u32>,
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new(json_state: JsValue) -> Result<Engine, JsValue> {
        // Initialize magic bitboards for O(1) slider attacks
        // crate::tiles::magic::init();

        let js_game: JsFullGame = serde_wasm_bindgen::from_value(json_state)?;

        // If this looks like a fresh game, clear any persistent search/TT state.
        if js_game.move_history.is_empty() && js_game.fullmove_number <= 1 {
            crate::search::reset_search_state();
        }

        let variant = js_game
            .variant
            .as_deref()
            .map(Variant::parse)
            .unwrap_or(Variant::Classical);

        // Apply world bounds from playableRegion if provided
        if let Some(wb) = &js_game.world_bounds {
            let left = wb.left.parse::<i64>().unwrap_or(-1_000_000_000_000_000);
            let right = wb.right.parse::<i64>().unwrap_or(1_000_000_000_000_000);
            let bottom = wb.bottom.parse::<i64>().unwrap_or(-1_000_000_000_000_000);
            let top = wb.top.parse::<i64>().unwrap_or(1_000_000_000_000_000);
            set_world_bounds(left, right, bottom, top);
        } else {
            // Always reset to defaults based on variant to prevent world borders from
            // leaking from previous games in the same worker (e.g. SPRT).
            let (left, right, bottom, top) = variant.get_default_bounds();
            set_world_bounds(left, right, bottom, top);
        }

        // Build starting GameState from JS board
        let mut board = Board::new();
        let mut white_has_royal = false;
        let mut black_has_royal = false;
        for p in &js_game.board.pieces {
            let x: i64 =
                p.x.parse()
                    .map_err(|_| JsValue::from_str("Invalid X coordinate"))?;
            let y: i64 =
                p.y.parse()
                    .map_err(|_| JsValue::from_str("Invalid Y coordinate"))?;

            let piece_type = p.piece_type.parse::<PieceType>().unwrap_or(PieceType::Pawn);

            let color = p
                .player
                .parse::<PlayerColor>()
                .unwrap_or(PlayerColor::White);

            if piece_type.is_royal() {
                match color {
                    PlayerColor::White => white_has_royal = true,
                    PlayerColor::Black => black_has_royal = true,
                    _ => {}
                }
            }

            board.set_piece(x, y, Piece::new(piece_type, color));
        }

        // Starting side (color that moved first) as reported by JS. The engine
        // will reconstruct the current side-to-move by replaying move_history.
        let js_turn = js_game
            .turn
            .parse::<PlayerColor>()
            .unwrap_or(PlayerColor::White);

        // Parse initial special rights (castling + pawn double-move)
        let mut special_rights = FxHashSet::default();
        for sr in js_game.special_rights {
            let parts: Vec<&str> = sr.split(',').collect();
            if parts.len() == 2
                && let (Ok(x), Ok(y)) = (parts[0].parse::<i64>(), parts[1].parse::<i64>())
            {
                special_rights.insert(Coordinate::new(x, y));
            }
        }

        // Parse en passant directly as i64 (used only when there is no move history)
        let parsed_en_passant = if let Some(ep) = js_game.en_passant {
            let sq_parts: Vec<&str> = ep.square.split(',').collect();
            let pawn_parts: Vec<&str> = ep.pawn_square.split(',').collect();

            if sq_parts.len() == 2 && pawn_parts.len() == 2 {
                if let (Ok(sq_x), Ok(sq_y), Ok(pawn_x), Ok(pawn_y)) = (
                    sq_parts[0].parse::<i64>(),
                    sq_parts[1].parse::<i64>(),
                    pawn_parts[0].parse::<i64>(),
                    pawn_parts[1].parse::<i64>(),
                ) {
                    Some(EnPassantState {
                        square: Coordinate::new(sq_x, sq_y),
                        pawn_square: Coordinate::new(pawn_x, pawn_y),
                    })
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Parse game rules from JS
        let game_rules = if let Some(js_rules) = js_game.game_rules {
            use game::{GameRules, PromotionRanks, WinCondition};

            let promotion_ranks = js_rules.promotion_ranks.map(|pr| PromotionRanks {
                white: pr
                    .white
                    .iter()
                    .filter_map(|s| s.parse::<i64>().ok())
                    .collect(),
                black: pr
                    .black
                    .iter()
                    .filter_map(|s| s.parse::<i64>().ok())
                    .collect(),
            });

            let (white_win_condition, black_win_condition) =
                if let Some(wc) = js_rules.win_conditions {
                    let parsed_white: Vec<WinCondition> =
                        wc.white.iter().filter_map(|s| s.parse().ok()).collect();
                    let parsed_black: Vec<WinCondition> =
                        wc.black.iter().filter_map(|s| s.parse().ok()).collect();

                    (
                        WinCondition::select(&parsed_white, black_has_royal),
                        WinCondition::select(&parsed_black, white_has_royal),
                    )
                } else {
                    (
                        if black_has_royal {
                            WinCondition::Checkmate
                        } else {
                            WinCondition::AllPiecesCaptured
                        },
                        if white_has_royal {
                            WinCondition::Checkmate
                        } else {
                            WinCondition::AllPiecesCaptured
                        },
                    )
                };

            let mut rules = GameRules {
                promotion_ranks: promotion_ranks.unwrap_or_else(|| PromotionRanks {
                    white: vec![],
                    black: vec![],
                }),
                promotion_types: None,
                promotions_allowed: js_rules.promotions_allowed,
                move_rule_limit: js_rules.move_rule,
                white_win_condition,
                black_win_condition,
            };
            rules.init_promotion_types();
            rules
        } else {
            game::GameRules::default()
        };

        // Precompute effective promotion ranks and dynamic back ranks once per
        // game from promotion_ranks. For standard chess this yields promo
        // ranks 8/1 and back ranks 1/8.
        let (white_promo_rank, black_promo_rank, white_back_rank, black_back_rank) = {
            let ranks = &game_rules.promotion_ranks;
            let white_promo = ranks
                .white
                .iter()
                .copied()
                .max()
                .unwrap_or(2_000_000_000_000_000);
            let black_promo = ranks
                .black
                .iter()
                .copied()
                .min()
                .unwrap_or(-2_000_000_000_000_000);

            // White's home side is near Black's promotion ranks, and vice versa.
            let wb = if black_promo == -2_000_000_000_000_000 {
                1
            } else {
                black_promo
            }; // white back rank
            let bb = if white_promo == 2_000_000_000_000_000 {
                8
            } else {
                white_promo
            }; // black back rank

            (white_promo, black_promo, wb, bb)
        };

        let spatial_indices = SpatialIndices::new(&board);

        let mut game = GameState {
            board,
            // Start with current turn; replayed history ensures correct side-to-move.
            turn: js_turn,
            special_rights,
            en_passant: None,
            halfmove_clock: js_game.halfmove_clock,
            fullmove_number: 1,
            material_score: 0,
            game_rules,
            variant: js_game.variant.as_deref().map(crate::Variant::parse),
            hash: 0, // Will be computed below
            hash_stack: Vec::with_capacity(js_game.move_history.len().saturating_add(8)),
            null_moves: 0,
            white_piece_count: 0,
            black_piece_count: 0,
            white_pawn_count: 0,
            black_pawn_count: 0,
            starting_white_pieces: 0,
            starting_black_pieces: 0,
            white_pieces: Vec::new(),
            black_pieces: Vec::new(),
            spatial_indices,
            starting_squares: FxHashSet::default(),
            white_back_rank,
            black_back_rank,
            white_promo_rank,
            black_promo_rank,
            white_king_pos: None,
            black_king_pos: None,
            check_squares_white: FxHashSet::default(),
            check_squares_black: FxHashSet::default(),
            slider_rays_white: [None; 8],
            slider_rays_black: [None; 8],
            discovered_check_squares_white: FxHashSet::default(),
            discovered_check_squares_black: FxHashSet::default(),
            pawn_hash: 0,
            white_nonpawn_hash: 0,
            black_nonpawn_hash: 0,
            minor_hash: 0,
            material_hash: 0,
            repetition: 0,
            white_non_pawn_material: false,
            black_non_pawn_material: false,
            effective_castling_rights: 0,
            castling_partner_counts: [0; 4],
            pinned_white: rustc_hash::FxHashMap::default(),
            pinned_black: rustc_hash::FxHashMap::default(),
            checkers_count_white: 0,
            checkers_count_black: 0,
            move_history: Vec::with_capacity(js_game.move_history.len().saturating_add(8)),
            plies_from_null: 0,
            total_phase: 0,
        };

        game.material_score = calculate_initial_material(&game.board);
        game.recompute_piece_counts(); // Rebuild piece lists and counts
        game.init_starting_piece_counts(); // Cache starting non-pawn piece counts for phase detection
        game.init_starting_squares();
        game.recompute_hash();

        // Helper to parse "x,y" into (i64, i64)
        fn parse_coords(coord_str: &str) -> Option<(i64, i64)> {
            let parts: Vec<&str> = coord_str.split(',').collect();
            if parts.len() != 2 {
                return None;
            }
            let x = parts[0].parse::<i64>().ok()?;
            let y = parts[1].parse::<i64>().ok()?;
            Some((x, y))
        }

        game.en_passant = parsed_en_passant;

        if js_game.move_history.is_empty() {
            game.turn = js_turn;
        } else {
            // Replay the full move history from the start position.
            // Like UCI: just apply moves directly by coordinates, no legal move generation needed.
            for hist in &js_game.move_history {
                if let (Some((from_x, from_y)), Some((to_x, to_y))) =
                    (parse_coords(&hist.from), parse_coords(&hist.to))
                {
                    let promo = hist.promotion.as_deref();
                    game.make_move_coords(from_x, from_y, to_x, to_y, promo);
                }
            }
        }

        // Optional clock information (similar to UCI wtime/btime/winc/binc).
        let clock = js_game.clock;
        let strength_level = js_game.strength_level;

        Ok(Engine {
            game,
            clock,
            strength_level,
        })
    }

    pub fn get_best_move(&mut self) -> JsValue {
        if let Some((best_move, _eval, _stats)) =
            search::get_best_move(&mut self.game, 50, u128::MAX, false, true)
        {
            let js_move = JsMove {
                from: format!("{},{}", best_move.from.x, best_move.from.y),
                to: format!("{},{}", best_move.to.x, best_move.to.y),
                promotion: best_move.promotion.map(|p| p.to_str().to_string()),
            };
            serde_wasm_bindgen::to_value(&js_move).unwrap()
        } else {
            JsValue::NULL
        }
    }

    #[cfg(feature = "eval_tuning")]
    #[wasm_bindgen]
    pub fn evaluate_with_features(&mut self) -> JsValue {
        crate::evaluation::reset_eval_features();
        let eval = crate::evaluation::evaluate(&self.game);
        let features = crate::evaluation::snapshot_eval_features();
        serde_wasm_bindgen::to_value(&JsEvalWithFeatures { eval, features }).unwrap()
    }

    /// Set search parameters from a JSON string.
    /// Only available when the `search_tuning` feature is enabled.
    /// Returns true on success, false on parse failure.
    #[cfg(feature = "search_tuning")]
    #[wasm_bindgen]
    pub fn set_search_params(&self, json: &str) -> bool {
        crate::search::params::set_search_params_from_json(json)
    }

    /// Get current search parameters as a JSON string.
    /// Only available when the `search_tuning` feature is enabled.
    #[cfg(feature = "search_tuning")]
    #[wasm_bindgen]
    pub fn get_search_params(&self) -> String {
        crate::search::params::get_search_params_as_json()
    }

    /// Derive an effective time limit for this move from the current clock and
    /// game state. Returns `(time_ms, is_soft_limit)`.
    ///
    /// **is_soft_limit** indicates whether the returned time is a suggestion
    /// (where exceeding it slightly is acceptable) vs a hard constraint.
    /// - Soft limit: untimed/infinite games with a suggested per-move limit
    /// - Hard limit: timed games where exceeding the budget risks flagging
    ///
    /// Uses logarithmic time formulas for proper scaling across
    /// different time controls (blitz, rapid, classical).
    fn effective_time_limit_ms(&self, requested_limit_ms: u32) -> (u128, u128, bool) {
        let Some(clock) = self.clock else {
            // No clock info: use the fixed per-move limit as a soft limit.
            // The search can use up to this time freely without flagging risk.
            let limit = requested_limit_ms as u128;
            return (limit, limit, true);
        };

        // Decide which side's clock to use.
        let (remaining_ms_raw, inc_ms_raw) = match self.game.turn {
            PlayerColor::White => (clock.wtime, clock.winc),
            PlayerColor::Black => (clock.btime, clock.binc),
            // Neutral side-to-move should not happen; fall back to
            // the requested limit as soft.
            PlayerColor::Neutral => {
                let limit = requested_limit_ms as u128;
                return (limit, limit, true);
            }
        };

        // If there is no usable clock information, use requested limit as soft.
        if remaining_ms_raw == 0 && inc_ms_raw == 0 {
            let limit = requested_limit_ms as u128;
            return (limit, limit, true);
        }

        // Handle zero remaining time with positive increment (increment-only)
        let remaining_ms = if remaining_ms_raw > 0 {
            remaining_ms_raw
        } else {
            // At least give ourselves a small buffer based on increment.
            inc_ms_raw.max(500)
        };

        let inc_ms = inc_ms_raw;

        // Dynamic Time Allocation
        // Calculates optimal and maximum thinking time based on remaining budget.

        // Move overhead for communication latency
        let move_overhead: u64 = 50;

        // Calculate scaled time (remaining time minus one move overhead)
        let scaled_time = remaining_ms.saturating_sub(move_overhead);

        // Maximum move horizon (centiMTG = moves to go * 100)
        // For games without movestogo, assume ~50 moves remaining
        // If less than 1 second, gradually reduce moves to go estimate
        let centi_mtg: i64 = if scaled_time >= 1000 {
            5051 // 50.51 moves * 100
        } else {
            ((scaled_time as f64) * 5.051) as i64
        };
        let centi_mtg = centi_mtg.max(100); // At least 1 move expected

        // timeLeft: total time we can use considering increment and overhead
        // Formula: remaining + inc * (MTG - 1) - overhead * (2 + MTG)
        // Division by 100 because centiMTG is in centimoves
        let time_left = (remaining_ms as i64
            + (inc_ms as i64 * (centi_mtg - 100) - move_overhead as i64 * (200 + centi_mtg)) / 100)
            .max(1) as f64;

        // original_time_adjust: logarithmic scaling factor based on available time
        // This factor prevents overspending at longer time controls by adjusting the
        // base allocation according to the total time budget.
        // For 10s games: log10(10000) = 4, adjust = 0.3128*4 - 0.4354 = 0.816
        // For 40m games: log10(2400000) = 6.38, adjust = 1.56
        let original_time_adjust = (0.3128 * time_left.max(1.0).log10() - 0.4354).max(0.1);

        // Log-based time constants for adaptive scaling
        let log_time_sec = (scaled_time as f64 / 1000.0).max(0.001).log10();

        // Optimum and maximum time constants
        let opt_constant = (0.0032116 + 0.000321123 * log_time_sec).min(0.00508017);
        let max_constant = (3.3977 + 3.0395 * log_time_sec).max(2.94761);

        // Game ply estimation based on fullmove number
        // In chess, typically ply = (fullmove_number - 1) * 2 + (is_black ? 1 : 0)
        let ply = self
            .game
            .fullmove_number
            .saturating_sub(1)
            .saturating_mul(2) as f64
            + if self.game.turn == PlayerColor::Black {
                1.0
            } else {
                0.0
            };

        // optScale: percentage of timeLeft to use for this move
        // Multiply by originalTimeAdjust to prevent overspending
        let opt_scale = ((0.0121431 + (ply + 2.94693_f64).powf(0.461073) * opt_constant)
            .min(0.213035 * remaining_ms as f64 / time_left))
            * original_time_adjust;

        // maxScale: multiplier from optimum to maximum
        let max_scale = max_constant.min(6.67704) + ply / 11.9847;

        // Calculate optimum and maximum allocations
        let optimum = (opt_scale * time_left) as u64;
        let maximum = ((max_scale * optimum as f64)
            .min(0.825179 * remaining_ms as f64 - move_overhead as f64)
            - 10.0)
            .max(0.0) as u64;

        // Special case: opening (first move) - limit to 5 seconds
        let mut maximum = maximum;
        let mut optimum = optimum;
        if self.game.fullmove_number == 1 && self.game.variant.is_some() {
            maximum = maximum.min(5000);
            optimum = optimum.min(5000);
        }

        // Apply sensible bounds
        let min_think_ms: u64 = 10;
        let optimum = optimum.max(min_think_ms);
        let maximum = maximum.max(optimum); // Maximum should never be less than optimum

        // Final safety cap: never exceed 82.5% of remaining time
        let absolute_cap = ((remaining_ms as f64) * 0.825 - move_overhead as f64) as u64;
        let optimum = optimum.min(absolute_cap.max(min_think_ms));
        let maximum = maximum.min(absolute_cap.max(min_think_ms));

        // Timed games are hard limits (risk of flagging).
        (optimum as u128, maximum as u128, false)
    }

    /// Timed search. This also exposes the search evaluation as an `eval` field alongside the move,
    /// so callers can reuse the same search for adjudication.
    /// thread_id is used for Lazy SMP - helper threads (id > 0) skip the first move.
    #[wasm_bindgen]
    pub fn get_best_move_with_time(
        &mut self,
        time_limit_ms: u32,
        silent: Option<bool>,
        max_depth: Option<usize>,
        noise_amp: Option<i32>,
        seed: Option<u64>,
    ) -> JsValue {
        // let legal_moves = self.game.get_legal_moves();
        // web_sys::console::log_1(&format!("Legal moves: {:?}", legal_moves).into());

        let (opt_time, max_time, is_soft_limit) = if time_limit_ms == 0 && max_depth.is_some() {
            // If explicit depth is requested with 0 time, treat as infinite time (fixed depth search)
            (u128::MAX, u128::MAX, true)
        } else {
            self.effective_time_limit_ms(time_limit_ms)
        };
        let silent = silent.unwrap_or(false);
        let depth = max_depth.unwrap_or(50).clamp(1, 50);
        let strength = self.strength_level;

        #[allow(unused_variables)]
        let pre_stats = crate::search::get_current_tt_stats();

        #[cfg(target_arch = "wasm32")]
        {
            if !silent {
                use crate::log;
                let variant = self
                    .game
                    .variant
                    .map_or("unknown".to_string(), |v| format!("{:?}", v));

                let tt_cap = pre_stats.tt_capacity;
                let tt_used = pre_stats.tt_used;
                let tt_fill = pre_stats.tt_fill_permille;

                if let Some(clock) = self.clock {
                    let side = match self.game.turn {
                        PlayerColor::White => "w",
                        PlayerColor::Black => "b",
                        PlayerColor::Neutral => "n",
                    };
                    log(&format!(
                        "info timealloc side {} wtime {} btime {} winc {} binc {} limit {} soft {} variant {} tt_cap {} tt_used {} tt_fill {}",
                        side,
                        clock.wtime,
                        clock.btime,
                        clock.winc,
                        clock.binc,
                        opt_time,
                        is_soft_limit,
                        variant,
                        tt_cap,
                        tt_used,
                        tt_fill,
                    ));
                } else {
                    log(&format!(
                        "info timealloc no_clock requested_limit {} effective_limit {} max_depth {:?} variant {} tt_cap {} tt_used {} tt_fill {}",
                        time_limit_ms, opt_time, max_depth, variant, tt_cap, tt_used, tt_fill,
                    ));
                }
            }
        }

        // Initialize randomness with global seed logic
        let effective_seed = seed.unwrap_or_else(get_random_seed);
        search::set_global_params(effective_seed, noise_amp);

        // Choose search path based on strength level.
        let (best_move, eval) = if strength.is_some_and(|s| s < 3) {
            // Use strength limited search (uses global seed we just set)
            if let Some((bm, ev, _stats)) = search::get_best_move_limited(
                &mut self.game,
                depth,
                opt_time,
                max_time,
                strength,
                silent,
                is_soft_limit,
            ) {
                (bm, ev)
            } else {
                return JsValue::NULL;
            }
        } else {
            // Normal search: use parallel version (handles both single and multi-threaded)
            if let Some((bm, ev, _stats)) = search::get_best_move_parallel(
                &mut self.game,
                depth,
                opt_time,
                max_time,
                silent,
                is_soft_limit,
            ) {
                (bm, ev)
            } else {
                return JsValue::NULL;
            }
        };

        let js_move = JsMoveWithEval {
            from: format!("{},{}", best_move.from.x, best_move.from.y),
            to: format!("{},{}", best_move.to.x, best_move.to.y),
            promotion: best_move.promotion.map(|p| p.to_str().to_string()),
            eval,
            depth,
        };
        serde_wasm_bindgen::to_value(&js_move).unwrap()
    }

    /// MultiPV-enabled timed search. Returns an array of PV lines (best moves with their
    /// evaluations and full PVs).
    ///
    /// Parameters:
    /// - `time_limit_ms`: Maximum time to think in milliseconds
    /// - `multi_pv`: Number of best moves to return (default 1). Must be >= 1.
    /// - `silent`: If true, suppress info output during search
    ///
    /// When `multi_pv` is 1, this has zero overhead compared to `get_best_move_with_time`.
    /// For `multi_pv` > 1, subsequent PV lines are found by re-searching the position
    /// with previously found best moves excluded.
    #[wasm_bindgen]
    pub fn get_best_moves_multipv(
        &mut self,
        time_limit_ms: u32,
        multi_pv: Option<usize>,
        silent: Option<bool>,
    ) -> JsValue {
        let (opt_time, max_time, is_soft_limit) = self.effective_time_limit_ms(time_limit_ms);
        let silent = silent.unwrap_or(false);
        let multi_pv = multi_pv.unwrap_or(1).max(1);

        let result = search::get_best_moves_multipv(
            &mut self.game,
            50,
            opt_time,
            max_time,
            multi_pv,
            silent,
            is_soft_limit,
        );

        // Convert to JS-friendly format
        let js_lines: Vec<JsPVLine> = result
            .lines
            .iter()
            .map(|line| {
                // Format PV as array of "x,y->x,y" strings
                let pv_strings: Vec<String> = line
                    .pv
                    .iter()
                    .map(|m| {
                        format!(
                            "{},{}->{},{}{}",
                            m.from.x,
                            m.from.y,
                            m.to.x,
                            m.to.y,
                            m.promotion.map_or("", |p| p.to_site_code())
                        )
                    })
                    .collect();

                JsPVLine {
                    from: format!("{},{}", line.mv.from.x, line.mv.from.y),
                    to: format!("{},{}", line.mv.to.x, line.mv.to.y),
                    promotion: line.mv.promotion.map(|p| p.to_str().to_string()),
                    eval: line.score,
                    depth: line.depth,
                    pv: pv_strings,
                }
            })
            .collect();

        serde_wasm_bindgen::to_value(&js_lines).unwrap_or(JsValue::NULL)
    }

    pub fn perft(&mut self, depth: usize) -> u64 {
        self.game.perft(depth)
    }

    /// Returns all legal moves as a JS array of {from: "x,y", to: "x,y", promotion: string|null}
    pub fn get_legal_moves_js(&mut self) -> JsValue {
        let pseudo_legal = self.game.get_legal_moves();
        let mut legal_moves: Vec<JsMove> = Vec::new();

        for m in pseudo_legal {
            let undo = self.game.make_move(&m);
            let illegal = self.game.is_move_illegal();
            self.game.undo_move(&m, undo);

            if !illegal {
                legal_moves.push(JsMove {
                    from: format!("{},{}", m.from.x, m.from.y),
                    to: format!("{},{}", m.to.x, m.to.y),
                    promotion: m.promotion.map(|p| p.to_str().to_string()),
                });
            }
        }

        serde_wasm_bindgen::to_value(&legal_moves).unwrap_or(JsValue::NULL)
    }

    /// Returns true if the current side to move is in check.
    pub fn is_in_check(&self) -> bool {
        self.game.is_in_check()
    }

    /// Returns true if either side has sufficient material to force checkmate.
    /// Returns false if the position is a dead draw due to insufficient material.
    /// This can be used by the SPRT harness to detect insufficient material draws.
    pub fn is_sufficient_material(&self) -> bool {
        // Use the evaluate_insufficient_material function
        // None = sufficient, Some(0) = dead draw, Some(n) = drawish
        match evaluation::insufficient_material::evaluate_insufficient_material(&self.game) {
            None => true,     // Sufficient material
            Some(0) => false, // Dead draw (insufficient)
            Some(_) => true,  // Drawish but not dead draw
        }
    }
}

impl Engine {
    /// Return the engine's static evaluation of the current position in centipawns,
    /// from the side-to-move's perspective (positive = advantage for side to move).
    pub fn evaluate_position(game: &GameState) -> i32 {
        #[cfg(feature = "nnue")]
        return crate::evaluation::evaluate(game, None);
        #[cfg(not(feature = "nnue"))]
        return crate::evaluation::evaluate(game);
    }
}
