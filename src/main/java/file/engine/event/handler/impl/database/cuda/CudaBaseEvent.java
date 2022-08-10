package file.engine.event.handler.impl.database.cuda;

import file.engine.event.handler.Event;

class CudaBaseEvent extends Event {

    CudaBaseEvent() {
        this.setBlock();
    }
}
