package file.engine.configs;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class AdvancedConfigEntity {

    private long waitForInputAndPrepareSearchTimeoutInMills;

    private long waitForInputAndStartSearchTimeoutInMills;

    private long clearIconCacheTimeoutInMills;
}
