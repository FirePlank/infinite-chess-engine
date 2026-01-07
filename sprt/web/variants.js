// Extract variant positions from the TypeScript variant definitions
// This handles timestamp-based position selection by picking the latest version

// Default variant positions (latest versions as of current time)
// Special rights are indicated by '+' suffix in ICN strings (e.g., "P1,2+")
const VARIANTS = {
    Classical: {
        position: 'r1,8+|n2,8|b3,8|q4,8|k5,8+|b6,8|n7,8|r8,8+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|R1,1+|N2,1|B3,1|Q4,1|K5,1+|B6,1|N7,1|R8,1+',
        game_rules: {
            promotions_allowed: ['q', 'r', 'b', 'n'],
        },
    },
    Confined_Classical: {
        position: 'P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|R1,1+|R8,1+|r1,8+|r8,8+|N2,1|N7,1|n2,8|n7,8|B3,1|B6,1|b3,8|b6,8|Q4,1|q4,8|K5,1+|k5,8+|ob0,0|ob0,1|ob0,2|ob0,7|ob0,8|ob0,9|ob9,0|ob9,1|ob9,2|ob9,7|ob9,8|ob9,9|ob1,0|ob2,0|ob3,0|ob4,0|ob5,0|ob6,0|ob7,0|ob8,0|ob1,9|ob2,9|ob3,9|ob4,9|ob5,9|ob6,9|ob7,9|ob8,9',
        game_rules: {
            promotions_allowed: ['q', 'r', 'b', 'n'],
        },
        hasCustomEval: true,
    },
    Classical_Plus: {
        position: 'p1,9+|p2,9+|p3,9+|p6,9+|p7,9+|p8,9+|p0,8+|r1,8+|n2,8|b3,8|q4,8|k5,8+|b6,8|n7,8|r8,8+|p9,8+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|p3,5+|p6,5+|P3,4+|P6,4+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P0,1+|R1,1+|N2,1|B3,1|Q4,1|K5,1+|B6,1|N7,1|R8,1+|P9,1+|P1,0+|P2,0+|P3,0+|P6,0+|P7,0+|P8,0+',
        game_rules: {
            promotions_allowed: ['q', 'r', 'b', 'n'],
        },
    },
    Core: {
        position: 'p-1,10+|p3,10+|p4,10+|p5,10+|p6,10+|p10,10+|p0,9+|p9,9+|n0,8|r1,8+|n2,8|b3,8|q4,8|k5,8+|b6,8|n7,8|r8,8+|n9,8|p-2,7+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|p11,7+|p-3,6+|p12,6+|p1,5+|P2,5+|P7,5+|p8,5+|P1,4+|p2,4+|p7,4+|P8,4+|P-3,3+|P12,3+|P-2,2+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P11,2+|N0,1|R1,1+|N2,1|B3,1|Q4,1|K5,1+|B6,1|N7,1|R8,1+|N9,1|P0,0+|P9,0+|P-1,-1+|P3,-1+|P4,-1+|P5,-1+|P6,-1+|P10,-1+',
        game_rules: {
            promotions_allowed: ['q', 'r', 'b', 'n'],
        },
    },
    CoaIP: {
        position: 'P-2,1+|P-1,2+|P0,2+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P9,2+|P10,2+|P11,1+|P-4,-6+|P-3,-5+|P-2,-4+|P-1,-5+|P0,-6+|P9,-6+|P10,-5+|P11,-4+|P12,-5+|P13,-6+|p-2,8+|p-1,7+|p0,7+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|p9,7+|p10,7+|p11,8+|p-4,15+|p-3,14+|p-2,13+|p-1,14+|p0,15+|p9,15+|p10,14+|p11,13+|p12,14+|p13,15+|HA-2,-6|HA11,-6|ha-2,15|ha11,15|R-1,1|R10,1|r-1,8|r10,8|CH0,1|CH9,1|ch0,8|ch9,8|GU1,1+|GU8,1+|gu1,8+|gu8,8+|N2,1|N7,1|n2,8|n7,8|B3,1|B6,1|b3,8|b6,8|Q4,1|q4,8|K5,1+|k5,8+',
        game_rules: {
            promotions_allowed: ['q', 'r', 'b', 'n', 'g', 'c', 'h'],
        },
    },
    CoaIP_HO: {
        position: 'p-4,14+|ha-2,14|p0,14+|p9,14+|ha11,14|p13,14+|p-3,13+|p-1,13+|p10,13+|p12,13+|p-2,12+|p11,12+|gu-1,9|hu0,9|ch1,9|ch8,9|hu9,9|gu10,9|p-1,8+|p0,8+|r1,8+|n2,8|b3,8|q4,8|k5,8+|b6,8|n7,8|r8,8+|p9,8+|p10,8+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P-1,1+|P0,1+|R1,1+|N2,1|B3,1|Q4,1|K5,1+|B6,1|N7,1|R8,1+|P9,1+|P10,1+|GU-1,0|HU0,0|CH1,0|CH8,0|HU9,0|GU10,0|P-2,-3+|P11,-3+|P-3,-4+|P-1,-4+|P10,-4+|P12,-4+|P-4,-5+|HA-2,-5|P0,-5+|P9,-5+|HA11,-5|P13,-5+',
        game_rules: {
            promotions_allowed: ['q', 'r', 'b', 'n', 'g', 'c', 'h', 'u'],
        },
    },
    CoaIP_RO: {
        position: 'P-2,1+|P-1,2+|P0,2+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P9,2+|P10,2+|P11,1+|P-4,-6+|P-3,-5+|P-2,-4+|P-1,-5+|P0,-6+|P9,-6+|P10,-5+|P11,-4+|P12,-5+|P13,-6+|p-2,8+|p-1,7+|p0,7+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|p9,7+|p10,7+|p11,8+|p-4,15+|p-3,14+|p-2,13+|p-1,14+|p0,15+|p9,15+|p10,14+|p11,13+|p12,14+|p13,15+|R-1,1|R10,1|r-1,8|r10,8|CH0,1|CH9,1|ch0,8|ch9,8|GU1,1+|GU8,1+|gu1,8+|gu8,8+|N2,1|N7,1|n2,8|n7,8|B3,1|B6,1|b3,8|b6,8|Q4,1|q4,8|K5,1+|k5,8+|RO-2,-6|RO11,-6|ro-2,15|ro11,15',
        game_rules: {
            promotions_allowed: ['q', 'r', 'b', 'n', 'g', 'c', 'o'],
        },
    },
    CoaIP_NO: {
        position: 'P-2,1+|P-1,2+|P0,2+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P9,2+|P10,2+|P11,1+|P-4,-6+|P-3,-5+|P-2,-4+|P-1,-5+|P0,-6+|P9,-6+|P10,-5+|P11,-4+|P12,-5+|P13,-6+|p-2,8+|p-1,7+|p0,7+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|p9,7+|p10,7+|p11,8+|p-4,15+|p-3,14+|p-2,13+|p-1,14+|p0,15+|p9,15+|p10,14+|p11,13+|p12,14+|p13,15+|R-1,1|R10,1|r-1,8|r10,8|CH0,1|CH9,1|ch0,8|ch9,8|GU1,1+|GU8,1+|gu1,8+|gu8,8+|N2,1|N7,1|n2,8|n7,8|B3,1|B6,1|b3,8|b6,8|Q4,1|q4,8|K5,1+|k5,8+|nr-2,16|nr11,16|NR-2,-7|NR11,-7',
        game_rules: {
            promotions_allowed: ['q', 'r', 'b', 'n', 'g', 'c', 's'],
        },
    },
    Palace: {
        position: 'K4,1|Q5,1|P6,2+|P5,2+|P4,2+|P3,2+|P2,2+|P1,2+|p1,4+|p2,4+|p3,4+|p4,4+|p5,4+|p6,4+|N6,1|AM3,1|Q2,1|N1,1|n1,5|n6,5|k4,5|q5,5|q2,5|am3,5|P6,-1+|P7,-1+|P8,-1+|P9,-1+|P1,-1+|P0,-1+|P-1,-1+|P-2,-1+|P2,-2+|P-3,-2+|P5,-2+|P10,-2+|p7,7+|p6,7+|p8,7+|p9,7+|p1,7+|p0,7+|p-1,7+|p-2,7+|p-3,8+|p2,8+|p5,8+|p10,8+|r-1,8|r-2,8|r8,8|r9,8|R8,-2|R9,-2|R-1,-2|R-2,-2|B0,-2|B1,-2|B7,-2|B6,-2|b0,8|b1,8|b7,8|b6,8',
        game_rules: {
            promotion_ranks: { white: ['4'], black: ['2'] },
            promotions_allowed: ['q', 'r', 'b', 'n', 'm'],
        },
    },
    Pawndard: {
        position: 'b4,14|b5,14|r4,12|r5,12|p2,10+|p3,10+|p6,10+|p7,10+|p1,9+|p8,9+|p0,8+|n2,8|n3,8|k4,8+|q5,8|n6,8|n7,8|p9,8+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|P1,5+|p2,5+|P3,5+|p6,5+|P7,5+|p8,5+|p1,4+|P2,4+|p3,4+|P6,4+|p7,4+|P8,4+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P0,1+|N2,1|N3,1|Q4,1|K5,1+|N6,1|N7,1|P9,1+|P1,0+|P8,0+|P2,-1+|P3,-1+|P6,-1+|P7,-1+|R4,-3|R5,-3|B4,-5|B5,-5',
        game_rules: {
            promotions_allowed: ['q', 'r', 'b', 'n'],
        },
    },
    Standarch: {
        position: 'p4,11+|p5,11+|p1,10+|p2,10+|p3,10+|p6,10+|p7,10+|p8,10+|p0,9+|ar4,9|ch5,9|p9,9+|p0,8+|r1,8+|n2,8|b3,8|q4,8|k5,8+|b6,8|n7,8|r8,8+|p9,8+|p0,7+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|p9,7+|P0,2+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P9,2+|P0,1+|R1,1+|N2,1|B3,1|Q4,1|K5,1+|B6,1|N7,1|R8,1+|P9,1+|P0,0+|AR4,0|CH5,0|P9,0+|P1,-1+|P2,-1+|P3,-1+|P6,-1+|P7,-1+|P8,-1+|P4,-2+|P5,-2+',
        game_rules: {
            promotions_allowed: ['q', 'r', 'b', 'n', 'c', 'a'],
        },
    },
    Space_Classic: {
        position: 'p-3,18+|r2,18|b4,18|b5,18|r7,18|p12,18+|p-4,17+|p13,17+|p-5,16+|p14,16+|p3,9+|p4,9+|p5,9+|p6,9+|n3,8|k4,8|q5,8|n6,8|p-6,7+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|p-8,6+|p-7,6+|p16,6+|p17,6+|p-9,5+|p18,5+|P-9,4+|P18,4+|P-8,3+|P-7,3+|P16,3+|P17,3+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P15,2+|N3,1|K4,1|Q5,1|N6,1|P3,0+|P4,0+|P5,0+|P6,0+|P-5,-7+|P14,-7+|P-4,-8+|P13,-8+|P-3,-9+|R2,-9|B4,-9|B5,-9|R7,-9|P12,-9+',
        game_rules: {
            promotions_allowed: ['q', 'r', 'b', 'n'],
        },
    },
    Space: {
        position: 'q4,31|ch4,23|p-12,18+|b4,18|p20,18+|p-11,17+|ar-10,17|p0,17+|b4,17|p8,17+|ar18,17|p19,17+|p-11,16+|p-10,16+|p-1,16+|p9,16+|p18,16+|p19,16+|p-1,15+|r0,15|ha4,15|r8,15|p9,15+|p3,6+|p4,6+|p5,6+|p2,5+|k4,5|p6,5+|n1,4|ce4,4|n7,4|p-10,3+|p-1,3+|p0,3+|p2,3+|p3,3+|p4,3+|p5,3+|p6,3+|p8,3+|p9,3+|p-12,2+|p-11,2+|p19,2+|p20,2+|p-13,1+|p21,1+|P-13,0+|P21,0+|P-12,-1+|P-11,-1+|P19,-1+|P20,-1+|P-1,-2+|P0,-2+|P2,-2+|P3,-2+|P4,-2+|P5,-2+|P6,-2+|P8,-2+|P9,-2+|P18,-2+|N1,-3|CE4,-3|N7,-3|P2,-4+|K4,-4|P6,-4+|P3,-5+|P4,-5+|P5,-5+|P-1,-14+|R0,-14|HA4,-14|R8,-14|P9,-14+|P-11,-15+|P-10,-15+|P-1,-15+|P9,-15+|P18,-15+|P19,-15+|P-11,-16+|AR-10,-16|P0,-16+|B4,-16|P8,-16+|AR18,-16|P19,-16+|P-12,-17+|B4,-17|P20,-17+|CH4,-22|Q4,-30',
        game_rules: {
            promotion_ranks: { white: ['4'], black: ['-3'] },
            promotions_allowed: ['q', 'r', 'b', 'n', 'h', 'e', 'a', 'c'],
        },
    },
    Abundance: {
        position: 'p-3,10+|ha-2,10|ha-1,10|r0,10|ha1,10|ha2,10|p3,10+|p-2,9+|p-1,9+|p1,9+|p2,9+|p-5,6+|gu-4,6|r-3,6+|b-2,6|b-1,6|k0,6+|b1,6|b2,6|r3,6+|gu4,6|p5,6+|p-4,5+|gu-3,5|n-1,5|q0,5|n1,5|gu3,5|p4,5+|p-3,4+|p-2,4+|gu-1,4|ch0,4|gu1,4|p2,4+|p3,4+|p-1,3+|p0,3+|p1,3+|P-1,-3+|P0,-3+|P1,-3+|P-3,-4+|P-2,-4+|GU-1,-4|CH0,-4|GU1,-4|P2,-4+|P3,-4+|P-4,-5+|GU-3,-5|N-1,-5|Q0,-5|N1,-5|GU3,-5|P4,-5+|P-5,-6+|GU-4,-6|R-3,-6+|B-2,-6|B-1,-6|K0,-6+|B1,-6|B2,-6|R3,-6+|GU4,-6|P5,-6+|P-2,-9+|P-1,-9+|P1,-9+|P2,-9+|P-3,-10+|HA-2,-10|HA-1,-10|R0,-10|HA1,-10|HA2,-10|P3,-10+',
        game_rules: {
            promotion_ranks: { white: ['6'], black: ['-6'] },
            promotions_allowed: ['q', 'r', 'b', 'n', 'g', 'h', 'c'],
        },
    },
    Pawn_Horde: {
        position: 'k5,2+|q4,2|r1,2+|n7,2|n2,2|r8,2+|b3,2|b6,2|P2,-1+|P3,-1+|P6,-1+|P7,-1+|P1,-2+|P2,-2+|P4,-2+|P5,-2+|P6,-2+|P7,-2+|P8,-2+|P1,-3+|P2,-3+|P4,-3+|P5,-3+|P6,-3+|P7,-3+|P8,-3+|P1,-4+|P2,-4+|P4,-4+|P5,-4+|P6,-4+|P7,-4+|P8,-4+|P1,-5+|P2,-5+|P4,-5+|P5,-5+|P6,-5+|P7,-5+|P8,-5+|P1,-6+|P2,-6+|P4,-6+|P5,-6+|P6,-6+|P7,-6+|P8,-6+|P3,-2+|P3,-3+|P3,-4+|P3,-5+|P3,-6+|P1,-7+|P2,-7+|P3,-7+|P4,-7+|P5,-7+|P6,-7+|P7,-7+|P8,-7+|P0,-6+|P0,-7+|P9,-6+|P9,-7+|p9,2+|p1,1+|p2,1+|p3,1+|p4,1+|p5,1+|p6,1+|p7,1+|p8,1+|p0,2+',
        game_rules: {
            promotion_ranks: { white: ['2'], black: ['-7'] },
            promotions_allowed: ['q', 'r', 'b', 'n'],
        },
        hasCustomEval: true,
    },
    Knightline: {
        position: 'k5,8|n3,8|n4,8|n6,8|n7,8|p-5,7+|p-4,7+|p-3,7+|p-2,7+|p-1,7+|p0,7+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|p9,7+|p10,7+|p11,7+|p12,7+|p13,7+|p14,7+|p15,7+|K5,1|N3,1|N4,1|N6,1|N7,1|P-5,2+|P-4,2+|P-3,2+|P-2,2+|P-1,2+|P0,2+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|P9,2+|P10,2+|P11,2+|P12,2+|P13,2+|P14,2+|P15,2+',
        game_rules: {
            promotions_allowed: ['q', 'r', 'b', 'n'],
        },
    },
    Obstocean: {
        position: 'ob-6,12|ob-5,12|ob-4,12|ob-3,12|ob-2,12|ob-1,12|ob0,12|ob1,12|ob2,12|ob3,12|ob4,12|ob5,12|ob6,12|ob7,12|ob8,12|ob9,12|ob10,12|ob11,12|ob12,12|ob13,12|ob14,12|ob15,12|ob-6,11|ob-5,11|ob-4,11|ob-3,11|ob-2,11|ob-1,11|ob0,11|ob1,11|ob2,11|ob3,11|ob4,11|ob5,11|ob6,11|ob7,11|ob8,11|ob9,11|ob10,11|ob11,11|ob12,11|ob13,11|ob14,11|ob15,11|ob-6,10|ob-5,10|ob-4,10|ob-3,10|ob-2,10|ob-1,10|ob0,10|ob1,10|ob2,10|ob3,10|ob4,10|ob5,10|ob6,10|ob7,10|ob8,10|ob9,10|ob10,10|ob11,10|ob12,10|ob13,10|ob14,10|ob15,10|ob-6,9|ob-5,9|ob-4,9|ob-3,9|ob-2,9|ob-1,9|ob0,9|ob1,9|ob2,9|ob3,9|ob4,9|ob5,9|ob6,9|ob7,9|ob8,9|ob9,9|ob10,9|ob11,9|ob12,9|ob13,9|ob14,9|ob15,9|ob-6,8|ob-5,8|ob-4,8|ob-3,8|ob-2,8|ob-1,8|ob0,8|r1,8+|n2,8|b3,8|q4,8|k5,8+|b6,8|n7,8|r8,8+|ob9,8|ob10,8|ob11,8|ob12,8|ob13,8|ob14,8|ob15,8|ob-6,7|ob-5,7|ob-4,7|ob-3,7|ob-2,7|ob-1,7|ob0,7|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|ob9,7|ob10,7|ob11,7|ob12,7|ob13,7|ob14,7|ob15,7|ob-6,6|ob-5,6|ob-4,6|ob-3,6|ob-2,6|ob-1,6|ob0,6|ob1,6|ob2,6|ob3,6|ob4,6|ob5,6|ob6,6|ob7,6|ob8,6|ob9,6|ob10,6|ob11,6|ob12,6|ob13,6|ob14,6|ob15,6|ob-6,5|ob-5,5|ob-4,5|ob-3,5|ob-2,5|ob-1,5|ob0,5|ob1,5|ob2,5|ob3,5|ob4,5|ob5,5|ob6,5|ob7,5|ob8,5|ob9,5|ob10,5|ob11,5|ob12,5|ob13,5|ob14,5|ob15,5|ob-6,4|ob-5,4|ob-4,4|ob-3,4|ob-2,4|ob-1,4|ob0,4|ob1,4|ob2,4|ob3,4|ob4,4|ob5,4|ob6,4|ob7,4|ob8,4|ob9,4|ob10,4|ob11,4|ob12,4|ob13,4|ob14,4|ob15,4|ob-6,3|ob-5,3|ob-4,3|ob-3,3|ob-2,3|ob-1,3|ob0,3|ob1,3|ob2,3|ob3,3|ob4,3|ob5,3|ob6,3|ob7,3|ob8,3|ob9,3|ob10,3|ob11,3|ob12,3|ob13,3|ob14,3|ob15,3|ob-6,2|ob-5,2|ob-4,2|ob-3,2|ob-2,2|ob-1,2|ob0,2|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|ob9,2|ob10,2|ob11,2|ob12,2|ob13,2|ob14,2|ob15,2|ob-6,1|ob-5,1|ob-4,1|ob-3,1|ob-2,1|ob-1,1|ob0,1|R1,1+|N2,1|B3,1|Q4,1|K5,1+|B6,1|N7,1|R8,1+|ob9,1|ob10,1|ob11,1|ob12,1|ob13,1|ob14,1|ob15,1|ob-6,0|ob-5,0|ob-4,0|ob-3,0|ob-2,0|ob-1,0|ob0,0|ob1,0|ob2,0|ob3,0|ob4,0|ob5,0|ob6,0|ob7,0|ob8,0|ob9,0|ob10,0|ob11,0|ob12,0|ob13,0|ob14,0|ob15,0|ob-6,-1|ob-5,-1|ob-4,-1|ob-3,-1|ob-2,-1|ob-1,-1|ob0,-1|ob1,-1|ob2,-1|ob3,-1|ob4,-1|ob5,-1|ob6,-1|ob7,-1|ob8,-1|ob9,-1|ob10,-1|ob11,-1|ob12,-1|ob13,-1|ob14,-1|ob15,-1|ob-6,-2|ob-5,-2|ob-4,-2|ob-3,-2|ob-2,-2|ob-1,-2|ob0,-2|ob1,-2|ob2,-2|ob3,-2|ob4,-2|ob5,-2|ob6,-2|ob7,-2|ob8,-2|ob9,-2|ob10,-2|ob11,-2|ob12,-2|ob13,-2|ob14,-2|ob15,-2|ob-6,-3|ob-5,-3|ob-4,-3|ob-3,-3|ob-2,-3|ob-1,-3|ob0,-3|ob1,-3|ob2,-3|ob3,-3|ob4,-3|ob5,-3|ob6,-3|ob7,-3|ob8,-3|ob9,-3|ob10,-3|ob11,-3|ob12,-3|ob13,-3|ob14,-3|ob15,-3',
        game_rules: {
            promotions_allowed: ['q', 'r', 'b', 'n'],
        },
        worldBorder: 0,
        hasCustomEval: true,
    },
    Chess: {
        position: 'r1,8+|n2,8|b3,8|q4,8|k5,8+|b6,8|n7,8|r8,8+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|R1,1+|N2,1|B3,1|Q4,1|K5,1+|B6,1|N7,1|R8,1+',
        game_rules: {
            promotions_allowed: ['q', 'r', 'b', 'n'],
        },
        worldBorder: 0,
        hasCustomEval: true,
    },
};

// Get variant position and special rights
function getVariantData(variantName) {
    const original = VARIANTS[variantName];
    if (!original) {
        throw new Error(`Unknown variant: ${variantName}`);
    }

    // Return a clone with defaults to avoid mutating the original VARIANTS object
    const variant = {
        ...original,
        game_rules: original.game_rules ? { ...original.game_rules } : {}
    };

    // Default promotion ranks if not set: White 8, Black 1
    if (!variant.game_rules.promotion_ranks) {
        variant.game_rules.promotion_ranks = { white: ['8'], black: ['1'] };
    }

    // Default move_rule: 100 halfmoves (50-move rule) for draw detection
    if (typeof variant.game_rules.move_rule !== 'number') {
        variant.game_rules.move_rule = 100;
    }

    return variant;
}

// Get all available variant names
function getAllVariants() {
    return Object.keys(VARIANTS);
}

// Get variants that have custom evaluation (volatility warning for SPRT)
function getVariantsWithCustomEval() {
    return Object.keys(VARIANTS).filter(name => VARIANTS[name].hasCustomEval === true);
}

export { VARIANTS, getVariantData, getAllVariants, getVariantsWithCustomEval };
