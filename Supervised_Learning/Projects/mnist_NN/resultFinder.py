cost = ['9.362209723802339', '2.6610893359442622', '3.134158189805267', '8.15769795974342', '28.69623676199511', '4.216978175765554', '4.77779668934433', '3.990903868947257', '3.323881688614026', '3.8322837552390263', '3.590031988399969', '3.2621856342610367', '3.8617277393123355', '3.312552865380782', '3.4643064655455573', '3.985981763121156', '3.3863487706782034', '3.2778242525170493', '3.5256101066470777', '3.4472079149238244', '3.249923892212703', '3.3511638390866407', '3.88292714435655', '3.3098333876803543', '3.522675854100405', '8.71998993689418', '3.5623419725823187', '8.297932741485411', '6.497544229423506', '3.502869047856571', '2.760105863457494', '11.5846525633353', '3.2504794225693856', '12.57930481688574', '3.2500590499842934', '3.349803545671947', '3.2554652369257795', '4.414753887140431', '6.308932404071091', '6.203737120636321', '3.3009947966259', '40.16279560882949', '3.2507299937651117', '3.608923489901501', '3.876319986436128', '3.5470527901202273', '3.269055889155917', '11.392318390729912', '6.039583947171679', '3.251311683226046', '12.007559190053417', '2.547058843562123', '5.263013395497348', '7.242823306531246', '8.861457486819575', '3.0555336485762727', '3.1558132109069343', '12.934314577580789', '3.25581805801398', '3.251163847603257', '3.2504121058656783', '3.2493209007146597', '145.06286096336805', '3.448265706495753', '4.284251169455716', '3.250701023635707', '3.2507273194398', '3.249356977213966', '3.2510892044410906', '3.2500757498412813', '3.2494538535135424', '3.254608380689447', 'nan', 'nan', '3.329084921752881', '8.101294586429072', '3.0617903003424147', '5.824337322823573', '3.2582220693720836', '3.893950249022156', '3.2493431004318136', '7.57912984423414', '2.9192607945281424', '3.2496832255555286', '3.8613940397879722', '3.2639496611062015', '3.2524993911742004', '3.2497572086658226', '3.250122438278872', '3.250361752834388', '3.2504085683395787', '3.1882059928857887', '4.899326508722375', '3.345152860888722', 'nan', '3.2507962228708442', 'nan', '3.2514537341974536', '3.6425623790658492', '3.254485673208886', '9.836032862840316', '2.441422167821436', '3.9306648744606787', '6.565787265443209', '2.628698224040172', '3.249342550397649', '3.694385155774276', '3.2497069251051642', '3.057375911477946', '3.24941339060741', '3.3314148601867837', '3.287084070667815', 'nan', '3.2511217251965308', 'nan', '3.2501278513388168', '3.591599402621432', 'nan', '3.2498618015376435', '3.2503609987938664', '3.251781216641567', '3.253067333049253', '3.2499902665340996', '3.806771842354749', 'nan', '10.258462996502915', '3.4860747260046514', '2.8928715751760143', '8.887467895447378', '4.372813772871866', '14.78825249307286', '5.295206279641899', '4.57343965919871', '10.404194224382296', '3.249274849895308', '3.249865960508751', '35.73249475725559', '3.2494189330930134', '3.2496543263762616', '3.252858452739255', '3.249485430664551', '3.2509822464213616', '3.250315787029599', '3.25035996600019', '3.2496937925061826', '3.2502958616280098', 'nan', '3.2508537209746176', '20.224591784693224', '3.673021969744012', '11.912915826179038', '3.591372003393689', '3.7274682420618777', '9.633438699345378', '5.264516854374568', '2.885802973459115', '2.8753372867297493', '3.3114248723646122', '4.270321980013244', '3.3054578082391473', '3.250717097849221', '4.94710818630986', 'nan', '3.2492909922246946', '3.2519621494475084', '3.250047465952931', '3.2501634329037308', '3.249895239439704', '3.2513245110666738', '15.282213036639051', '3.2508000367533585', '7.185101536995731', '12.830845509469563', '5.074123209412671']
cost = [float(x) for x in cost]
hyper = ['Lambda:0.0alpha:0.01', 'Lambda:0.0alpha:0.21', 'Lambda:0.0alpha:0.41', 'Lambda:0.0alpha:0.61', 'Lambda:0.0alpha:0.81', 'Lambda:0.0alpha:1.01', 'Lambda:0.0alpha:1.21', 'Lambda:0.0alpha:1.41', 'Lambda:0.0alpha:1.61', 'Lambda:0.0alpha:1.81', 'Lambda:0.0alpha:2.01', 'Lambda:0.0alpha:2.21', 'Lambda:0.0alpha:2.41', 'Lambda:0.0alpha:2.61', 'Lambda:0.0alpha:2.81', 'Lambda:0.0alpha:3.01', 'Lambda:0.0alpha:3.21', 'Lambda:0.0alpha:3.41', 'Lambda:0.0alpha:3.61', 'Lambda:0.0alpha:3.81', 'Lambda:0.0alpha:4.01', 'Lambda:0.0alpha:4.21', 'Lambda:0.0alpha:4.41', 'Lambda:0.0alpha:4.61', 'Lambda:0.0alpha:4.81', 'Lambda:0.5alpha:0.01', 'Lambda:0.5alpha:0.21', 'Lambda:0.5alpha:0.41', 'Lambda:0.5alpha:0.61', 'Lambda:0.5alpha:0.81', 'Lambda:0.5alpha:1.01', 'Lambda:0.5alpha:1.21', 'Lambda:0.5alpha:1.41', 'Lambda:0.5alpha:1.61', 'Lambda:0.5alpha:1.81', 'Lambda:0.5alpha:2.01', 'Lambda:0.5alpha:2.21', 'Lambda:0.5alpha:2.41', 'Lambda:0.5alpha:2.61', 'Lambda:0.5alpha:2.81', 'Lambda:0.5alpha:3.01', 'Lambda:0.5alpha:3.21', 'Lambda:0.5alpha:3.41', 'Lambda:0.5alpha:3.61', 'Lambda:0.5alpha:3.81', 'Lambda:0.5alpha:4.01', 'Lambda:0.5alpha:4.21', 'Lambda:0.5alpha:4.41', 'Lambda:0.5alpha:4.61', 'Lambda:0.5alpha:4.81', 'Lambda:1.0alpha:0.01', 'Lambda:1.0alpha:0.21', 'Lambda:1.0alpha:0.41', 'Lambda:1.0alpha:0.61', 'Lambda:1.0alpha:0.81', 'Lambda:1.0alpha:1.01', 'Lambda:1.0alpha:1.21', 'Lambda:1.0alpha:1.41', 'Lambda:1.0alpha:1.61', 'Lambda:1.0alpha:1.81', 'Lambda:1.0alpha:2.01', 'Lambda:1.0alpha:2.21', 'Lambda:1.0alpha:2.41', 'Lambda:1.0alpha:2.61', 'Lambda:1.0alpha:2.81', 'Lambda:1.0alpha:3.01', 'Lambda:1.0alpha:3.21', 'Lambda:1.0alpha:3.41', 'Lambda:1.0alpha:3.61', 'Lambda:1.0alpha:3.81', 'Lambda:1.0alpha:4.01', 'Lambda:1.0alpha:4.21', 'Lambda:1.0alpha:4.41', 'Lambda:1.0alpha:4.61', 'Lambda:1.0alpha:4.81', 'Lambda:1.5alpha:0.01', 'Lambda:1.5alpha:0.21', 'Lambda:1.5alpha:0.41', 'Lambda:1.5alpha:0.61', 'Lambda:1.5alpha:0.81', 'Lambda:1.5alpha:1.01', 'Lambda:1.5alpha:1.21', 'Lambda:1.5alpha:1.41', 'Lambda:1.5alpha:1.61', 'Lambda:1.5alpha:1.81', 'Lambda:1.5alpha:2.01', 'Lambda:1.5alpha:2.21', 'Lambda:1.5alpha:2.41', 'Lambda:1.5alpha:2.61', 'Lambda:1.5alpha:2.81', 'Lambda:1.5alpha:3.01', 'Lambda:1.5alpha:3.21', 'Lambda:1.5alpha:3.41', 'Lambda:1.5alpha:3.61', 'Lambda:1.5alpha:3.81', 'Lambda:1.5alpha:4.01', 'Lambda:1.5alpha:4.21', 'Lambda:1.5alpha:4.41', 'Lambda:1.5alpha:4.61', 'Lambda:1.5alpha:4.81', 'Lambda:2.0alpha:0.01', 'Lambda:2.0alpha:0.21', 'Lambda:2.0alpha:0.41', 'Lambda:2.0alpha:0.61', 'Lambda:2.0alpha:0.81', 'Lambda:2.0alpha:1.01', 'Lambda:2.0alpha:1.21', 'Lambda:2.0alpha:1.41', 'Lambda:2.0alpha:1.61', 'Lambda:2.0alpha:1.81', 'Lambda:2.0alpha:2.01', 'Lambda:2.0alpha:2.21', 'Lambda:2.0alpha:2.41', 'Lambda:2.0alpha:2.61', 'Lambda:2.0alpha:2.81', 'Lambda:2.0alpha:3.01', 'Lambda:2.0alpha:3.21', 'Lambda:2.0alpha:3.41', 'Lambda:2.0alpha:3.61', 'Lambda:2.0alpha:3.81', 'Lambda:2.0alpha:4.01', 'Lambda:2.0alpha:4.21', 'Lambda:2.0alpha:4.41', 'Lambda:2.0alpha:4.61', 'Lambda:2.0alpha:4.81', 'Lambda:2.5alpha:0.01', 'Lambda:2.5alpha:0.21', 'Lambda:2.5alpha:0.41', 'Lambda:2.5alpha:0.61', 'Lambda:2.5alpha:0.81', 'Lambda:2.5alpha:1.01', 'Lambda:2.5alpha:1.21', 'Lambda:2.5alpha:1.41', 'Lambda:2.5alpha:1.61', 'Lambda:2.5alpha:1.81', 'Lambda:2.5alpha:2.01', 'Lambda:2.5alpha:2.21', 'Lambda:2.5alpha:2.41', 'Lambda:2.5alpha:2.61', 'Lambda:2.5alpha:2.81', 'Lambda:2.5alpha:3.01', 'Lambda:2.5alpha:3.21', 'Lambda:2.5alpha:3.41', 'Lambda:2.5alpha:3.61', 'Lambda:2.5alpha:3.81', 'Lambda:2.5alpha:4.01', 'Lambda:2.5alpha:4.21', 'Lambda:2.5alpha:4.41', 'Lambda:2.5alpha:4.61', 'Lambda:2.5alpha:4.81', 'Lambda:3.0alpha:0.01', 'Lambda:3.0alpha:0.21', 'Lambda:3.0alpha:0.41', 'Lambda:3.0alpha:0.61', 'Lambda:3.0alpha:0.81', 'Lambda:3.0alpha:1.01', 'Lambda:3.0alpha:1.21', 'Lambda:3.0alpha:1.41', 'Lambda:3.0alpha:1.61', 'Lambda:3.0alpha:1.81', 'Lambda:3.0alpha:2.01', 'Lambda:3.0alpha:2.21', 'Lambda:3.0alpha:2.41', 'Lambda:3.0alpha:2.61', 'Lambda:3.0alpha:2.81', 'Lambda:3.0alpha:3.01', 'Lambda:3.0alpha:3.21', 'Lambda:3.0alpha:3.41', 'Lambda:3.0alpha:3.61', 'Lambda:3.0alpha:3.81', 'Lambda:3.0alpha:4.01', 'Lambda:3.0alpha:4.21', 'Lambda:3.0alpha:4.41', 'Lambda:3.0alpha:4.61']
print(min(cost), hyper[cost.index(min(cost))])
print(len(cost))
