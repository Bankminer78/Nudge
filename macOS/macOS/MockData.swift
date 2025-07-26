
import Foundation

struct Update: Identifiable {
    let id = UUID()
    let title: String
    let description: String
}

class MockData {
    private static var checkCount = 0
    
    static func checkForUpdates(completion: @escaping ([Update]?) -> Void) {
        checkCount += 1
        
        if checkCount >= 10 {
            checkCount = 0
            let updates = [
                Update(title: "New Feature", description: "Check out the latest feature we've added."),
                Update(title: "Bug Fixes", description: "We've squashed some bugs to improve your experience.")
            ]
            completion(updates)
        } else {
            completion(nil)
        }
    }
}
