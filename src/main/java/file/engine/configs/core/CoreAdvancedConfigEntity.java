package file.engine.configs.core;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class CoreAdvancedConfigEntity {

    private long searchWarmupTimeoutInMills;

    private long waitForSearchTasksTimeoutInMills;

    private boolean isDeleteUsnOnExit;

    private long restartMonitorDiskThreadTimeoutInMills;
}
