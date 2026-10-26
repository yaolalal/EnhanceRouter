from common import *
setup()
write(RUN/'PROTOCOL_AMENDMENT_02.json',dict(before_any_dev_router_fit_or_evaluation=True,
 D3='Stratified k/class validation sampled from non-training nodes excluding fixed1000; cap to all available where class pool has fewer than k; record actual counts. Original random500 subset cannot achieve k/class on long-tail data.',
 D3_sampling_seed='devseed+1181',original_small_outputs='pilot only, excluded from final D3',
 B0_support_shortage='Support sampling likewise retains at most k when official train pool has fewer; exact IDs are registered.',metadata=metadata()))
