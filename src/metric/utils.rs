use std::collections::BinaryHeap;
use crate::pqfs::SearchResult;

pub fn smallest_heap_with_existing<I>(k: usize, iter: I, heap: &mut BinaryHeap<SearchResult>)
where
    I: Iterator<Item = SearchResult>,
{
    for item in iter {
        if heap.len() < k {
            heap.push(item);
        } else if let Some(mut root) = heap.peek_mut() {
            if item < *root {
                *root = item;
            }
        }
    }
}