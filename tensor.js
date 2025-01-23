function TENSOR(fill,shapearr) {

  return maketensor(shapearr.length,shapearr,fill);

}

function SHAPE(ARR1) {

  let shapeARR = [ARR1.length];
	for (ga = 0; ga < 6; ga++) {
    let testv = dimen(false,ARR1,maketensor(1,[ga+2],0));
		if (testv === undefined) {
			break;
		}
	  shapeARR[ga+1] = testv.length;
	}
  return shapeARR;

}

function dimen(assign,arr,parr,val) {
	let p0 = parr[0];
	let p1 = parr[1];
	let p2 = parr[2];
	let p3 = parr[3];
	let p4 = parr[4];
	let p5 = parr[5];
	let dst = parr.length;
	if (assign == true) {
		if (dst == 1) {
			arr[p0] = val;
		}//1D
		else if (dst == 2) {
			arr[p0][p1] = val;
		}//2D
		else if (dst == 3) {
			arr[p0][p1][p2] = val;
		}//3D
		else if (dst == 4) {
			arr[p0][p1][p2][p3] = val;
		}//4D
		else if (dst == 5) {
			arr[p0][p1][p2][p3][p4] = val;
		}//5D
		else if (dst == 6) {
			arr[p0][p1][p2][p3][p4][p5] = val;
		}//6D
	}
	else {
		if (dst == 1) {
			return arr[p0];
		}//1D
		else if (dst == 2) {
			return arr[p0][p1];
		}//2D
		else if (dst == 3) {
			return arr[p0][p1][p2];
		}//3D
		else if (dst == 4) {
			return arr[p0][p1][p2][p3];
		}//4D
		else if (dst == 5) {
			return arr[p0][p1][p2][p3][p4];
		}//5D
		else if (dst == 6) {
			return arr[p0][p1][p2][p3][p4][p5];
		}//6D
	}
}//different dimensional arrays- assign: bool- t:assign or f:return

function maketensor(dim,shapeARR,fill,ifrand,randl,randh,ifroundrand,ascending) {
	
	let ra = []
	for (g = 0; g < shapeARR[0] && dim > 0; g++) {
		ra[g] = [];
		for (g1 = 0; g1 < shapeARR[1] && dim > 1; g1++) {
			ra[g][g1] = [];
			for (g2 = 0; g2 < shapeARR[2] && dim > 2; g2++) {
				ra[g][g1][g2] = [];
				for (g3 = 0; g3 < shapeARR[3] && dim > 3; g3++) {
					ra[g][g1][g2][g3] = [];
					for (g4 = 0; g4 < shapeARR[4] && dim > 4; g4++) {
						ra[g][g1][g2][g3][g4] = [];
						for (g5 = 0; g5 < shapeARR[5] && dim > 5; g5++) {
							ra[g][g1][g2][g3][g4][g5] = getfill([g,g1,g2,g3,g4,g5]);
						}
						if (dim == 5) {
							ra[g][g1][g2][g3][g4] = getfill([g,g1,g2,g3,g4]);
						}
					}
					if (dim == 4) {
						ra[g][g1][g2][g3] = getfill([g,g1,g2,g3]);
					}
				}
				if (dim == 3) {
					ra[g][g1][g2] = getfill([g,g1,g2]);
				}
			}
			if (dim == 2) {
				ra[g][g1] = getfill([g,g1]);
			}
		}
		if (dim == 1) {
			ra[g] = getfill([g]);
		}
	}//initializes arrays
	
	function getfill(parr) {
		if (ifrand == true) {
			if (ifroundrand == true) {
				return rr(randl,randh+1);
			}
			else {
				return random(randl,randh);
			}
		}
		else if (ascending == true) {
			return parr[parr.length-1];
		}
		else if (typeof fill === 'function') {
			return fill(parr);
		}
		else if (typeof fill === 'object') {
			return CA(fill);
		}
		else {
			return fill;
		}
	}
	
	return ra;
	
}//limit of 6 dimensions, randl = lower bound, randh = upper bound
