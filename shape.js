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
