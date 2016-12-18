#### Description of non-product variables [N/Y]: [not added/added]
- Customer activity indicators
	- ind_empleado[N]: Employee index: A active, B ex employed, F filial, N not employee, P pasive
	- ind_nuevo[N]: New customer Index. 1 if the customer registered in the last 6 months.
	- antiguedad[N]: Customer seniority (in months) [need to be rescaled and binned if added]
	- tiprel_1mes[N]: Customer relation type at the beginning of the month, A (active), I (inactive), P (former customer),R (Potential)
	- canal_entrada[N]: channel used by the customer to join[Big categorical, need to combine many levels before added]
	- ind_actividad_cliente[Y]: Activity index (1, active customer; 0, inactive customer)
	- indrel[Y]: 1 (First/Primary), 99 (Primary customer during the month but not at the end of the month)
	- indrel_1mes[Y]: Customer type at the beginning of the month ,1 (First/Primary customer), 2 (co-owner ),P (Potential),3 (former primary), 4(former co-owner)
	- segmento[Y]: segmentation: 01 - VIP, 02 - Individuals 03 - college graduated

- Customer personal profile indicators
	- sexo[N]: Customer's sex
	- indfall[N]: Deceased index. N/S
	- renta[Y]: Gross income of the household
	- age[Y]: Age
	- indext[Y]: Foreigner index (S (Yes) or N (No) if the customer's birth country is different than the bank country)


- Customer geographical location indicators
	- indresi[N]: Residence index (S (Yes) or N (No) if the residence country is the same than the bank country)[Geo]
	- pais_residencia[N]: Customer's Country residence [Not important][Geo]
	- nomprov[N]: Province name[Geo]

- Other
	- month[Y]: month of the year in which the customer exists [1,2,...,12]