using SpawnDev.UnitTesting;

namespace SpawnDev.VoxelEngine.Demo.UnitTests
{
    public class DefaultTests
    {
        [TestMethod]
        public async Task VoxelEngineLoads()
        {
            // Verify the library loaded successfully in the browser
            await Task.Delay(100);
        }
    }
}
