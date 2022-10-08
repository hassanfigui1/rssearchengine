let star = document.querySelectorAll('input[type="radio"]');
let showValue = document.querySelector('input[name="review"]');
		
	for (let i = 0; i < star.length; i++) {
			star[i].addEventListener('click', function() {
				i = this.value;
		
				showValue.value = i 
			});
}




