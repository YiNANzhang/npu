use npug::reader::GraphReader;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: npug-inspect <file.npug>");
        std::process::exit(2);
    });
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let r = match GraphReader::from_bytes(&bytes) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("parse error: {e}");
            std::process::exit(1);
        }
    };

    println!("file     : {path} ({} bytes)", bytes.len());
    println!("abi      : 0x{:06x}", r.abi_version());
    println!("target   : {:?}", r.target());
    println!("producer : {}", r.producer());
    println!();

    let tensors = r.tensors();
    println!("tensors  : {}", tensors.len());
    for (i, t) in tensors.iter().enumerate() {
        print!("  [{i:3}] {:<32} {:?} {:?}", t.name, t.dtype, t.dims);
        if t.quant.scheme != npug::QuantScheme::None {
            print!(" quant={:?}/blk{}", t.quant.scheme, t.quant.block_size);
        }
        if t.region != npug::MemoryRegion::Unknown {
            print!(" region={:?}@0x{:x}", t.region, t.offset);
        }
        println!();
    }

    let kernels = r.kernels();
    println!("kernels  : {}", kernels.len());
    for (i, k) in kernels.iter().enumerate() {
        println!("  [{i:3}] {:<32} {:?} buf={} entry=0x{:x}",
                 k.name, k.kind, k.buffer, k.entry_offset);
    }

    let eps = r.entry_points_full();
    println!("entry pts: {}", eps.len());
    for (i, ep) in eps.iter().enumerate() {
        println!("  [{i:3}] {:<16} in={:?} out={:?} buckets={}",
                 ep.name, ep.inputs, ep.outputs, ep.buckets.len());
        for (bi, b) in ep.buckets.iter().enumerate() {
            println!("        bucket[{bi}] shape={:?} schedule={}",
                     b.shape_hint_dims, b.schedule.len());
        }
    }
}
