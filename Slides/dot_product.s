
for(i = 0; i < 40; i++)
{
    y += c[i]*x[i];
}

MVK     .S1  20,  B0

LOOP:
    LDW      .D1  *A4++,  A5
    LDW      .D2  *B4++,  B5
    NOP       3
    MPY      .M1X A5,  B5,  A6
    MPYH     .M2X A5,  B5,  B6
    ADD      .L1  A7,  A6,  A7
    ADD      .L2  B7,  B6,  B7
    [B0]SUB  .S1  B0,  1,  B0
    [B0]B    .S2  LOOP








SOFTWARE PIPELINED CODE:


    LDW        .D1  *A4++,  A5
    || LDW     .D2  *B4++,  B5

    LDW        .D1  *A4++,  A5
    || LDW     .D2  *B4++,  B5
    || [B0]SUB .S1  B0,  1,  B0

    LDW        .D1  *A4++,  A5
    || LDW     .D2  *B4++,  B5
    || [B0]SUB .S1  B0,  1,  B0
    || [B0]B   .S2  LOOP

    LDW        .D1  *A4++,  A5
    || LDW     .D2  *B4++,  B5
    || [B0]SUB .S1  B0,  1,  B0
    || [B0]B   .S2  LOOP

    LDW        .D1  *A4++,  A5
    || LDW     .D2  *B4++,  B5
    || [B0]SUB .S1  B0,  1,  B0
    || [B0]B   .S2  LOOP

    LDW        .D1  *A4++,  A5
    || LDW     .D2  *B4++,  B5
    || [B0]SUB .S1  B0,  1,  B0
    || [B0]B   .S2  LOOP
    || MPY     .M1X A5,  B5,  A6
    || MPYH    .M2X A5,  B5,  B6

    LDW        .D1  *A4++,  A5
    || LDW     .D2  *B4++,  B5
    || [B0]SUB .S1  B0,  1,  B0
    || [B0]B   .S2  LOOP
    || MPY     .M1X A5,  B5,  A6
    || MPYH    .M2X A5,  B5,  B6

LOOP:
    LDW        .D1  *A4++,  A5
    || LDW     .D2  *B4++,  B5
    || [B0]SUB .S1  B0,  1,  B0
    || [B0]B   .S2  LOOP
    || MPY     .M1X A5,  B5,  A6
    || MPYH    .M2X A5,  B5,  B6
    || ADD     .L1  A7,  A6,  A7
    || ADD     .L2  B7,  B6,  B7