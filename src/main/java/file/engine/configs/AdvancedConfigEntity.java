package file.engine.configs;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class AdvancedConfigEntity {

    private long waitForInputAndPrepareSearchTimeoutInMills;

    private long waitForInputAndStartSearchTimeoutInMills;

    private long clearIconCacheTimeoutInMills;
}
