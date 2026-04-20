use crate::generated::npug as fb;
use crate::version;
use flatbuffers::FlatBufferBuilder;

pub struct GraphBuilder<'a> {
    fbb: FlatBufferBuilder<'a>,
    producer: Option<String>,
    target: fb::TargetId,
}

impl<'a> GraphBuilder<'a> {
    pub fn new() -> Self {
        Self {
            fbb: FlatBufferBuilder::with_capacity(1024),
            producer: None,
            target: fb::TargetId::AutoSocV1,
        }
    }

    pub fn set_producer(&mut self, s: &str) -> &mut Self {
        self.producer = Some(s.to_string());
        self
    }

    pub fn set_target(&mut self, t: fb::TargetId) -> &mut Self {
        self.target = t;
        self
    }

    pub fn finish(mut self) -> Vec<u8> {
        let producer = self.producer.take().map(|s| self.fbb.create_string(&s));
        let entry_points = self.fbb.create_vector::<flatbuffers::WIPOffset<fb::EntryPoint>>(&[]);

        let graph = fb::Graph::create(
            &mut self.fbb,
            &fb::GraphArgs {
                abi_version: fb::AbiVersion(version::CURRENT),
                target: self.target,
                producer,
                entry_points: Some(entry_points),
            },
        );

        self.fbb.finish(graph, Some(std::str::from_utf8(version::MAGIC).unwrap()));
        self.fbb.finished_data().to_vec()
    }
}

impl<'a> Default for GraphBuilder<'a> {
    fn default() -> Self { Self::new() }
}
