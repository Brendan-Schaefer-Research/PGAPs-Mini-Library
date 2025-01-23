function ACTIVATE(ARR) {

  let ra = [];
  for (g = 0; g < ARR.length; g++) {
	  ra[g] = (1/(1+(pow(e,-1*scale*ARR[g]))));
  }
  return ra;
	
}
