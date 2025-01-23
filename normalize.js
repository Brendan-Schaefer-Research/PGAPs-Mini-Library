function NORMALIZE(ARR,scalar) {
	
	let arrneg = mult2d([CA(ARR)],[maketensor(1,[ARR.length],-1)])[0];
	let nv = max(max(ARR),max(arrneg))/scalar;//find maximum value
	return div2d([CA(ARR)],[maketensor(1,[ARR.length],nv)])[0];
	
}
