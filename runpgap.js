function runPGAP(input) {

	let last = [];
	for (ii = 0; ii < learningset; ii++) {
		last[ii] = CA(encoders[input[ii]]);
	}//vector encoding
	
	for (ii = 0; ii < iterations; ii++) {
		opxd("add",last[ii],identities[ii]);
		opxd("add",last[ii],maketensor(1,[encodesize],positioners(ii+1)));
	}//identities and pos encoding
	
	for (ll = 0; ll < layers; ll++) {
		//run attention network
		let kprod = [];
		for (ii = 0; ii < iterations; ii++) {
			//get attention and result
			let linout = runlinear(CA(last[ii]),aweights[ll][ii].length,false,aweights[ll][ii],abiases[ll][ii]);//get linear output
			opxd("sub",linout[0],maketensor(1,[linout[0].length],1/2));//scale entire last layer to avg 0
			last[ii] = CA(linout[0]).slice(0,encodesize);//add first part to last
			
			//store values
			neuronstore[ll][ii] = linout[1];
			kprod[ii] = CA(linout[0]).slice(encodesize,2*encodesize);
		}
		
		//run connector network
		let attenout = runlinear(concatenate(kprod),1,false,[apweights[ll]],[apbiases[ll]]);
		let sumkprod = opxd("sub",attenout[0],maketensor(1,[attenout[0].length],1/2));//scale entire last layer to avg 0
		apneur[ll] = attenout[1];

		for (ii = 0; ii < iterations; ii++) {
			opxd("add",last[ii],sumkprod.slice(ii*encodesize,(ii+1)*encodesize));
		}//add corresponding section of attention to last
		
	}//pgap
	
	//calculate final probabilities
	let avglast = maketensor(1,[encodesize],0);
	for (a = 0; a < learningset; a++) {
		opxd("add",avglast,last[a]);
	}//sum
	opxd("div",avglast,maketensor(1,[encodesize],learningset));//divide by learningset
	let encout = runlinear(CA(avglast),1,false,[transpose(CA(encoders))],[]);
	encneur = encout[1];
	
	return encout[0];
	
}
