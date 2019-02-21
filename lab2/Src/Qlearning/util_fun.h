#pragma once
/*************************************************************************
/* ECE 285: GPU Programmming 2019 Winter quarter
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#define SCEIL(x,n) ((x + (1 << n) - 1) >> n)
#define XCEIL(x,n) ( SCEIL(x,n) << n)