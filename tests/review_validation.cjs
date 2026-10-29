const assert=require('node:assert/strict');
const {validateReview}=require('../review_tool/review.js');
const valid={intent:'recommend',reviewer_id:'R01',family_id:'f1',entities:[{mention:'silk',value:'silk',negated:false}],slots:{occasion:'office'},remove_slots:[]};
assert.equal(validateReview(valid,'Use silk',['silk']),true);
for(const patch of [{intent:'invented'},{reviewer_id:''},{family_id:null},{entities:null},{slots:[]},{remove_slots:null},
 {entities:[{mention:'cotton',value:'silk',negated:false}]},{entities:[{mention:'silk',value:'silk',negated:'false'}]},
 {slots:{budget_lkr:Infinity}},{slots:{occasion:'moon'}},{remove_slots:['made_up']}]){
 assert.throws(()=>validateReview({...valid,...patch},'Use silk',['silk']));
}
assert.throws(()=>validateReview(valid,'silk or silk',['silk']));
assert.equal(validateReview({...valid,entities:[{mention:'silk',value:'silk',negated:false,start:8}]},'silk or silk',['silk']),true);
console.log('Review label/type/slot/span/negation/repeated-mention validation passed.');
