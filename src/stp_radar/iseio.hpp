#include <vector>
#include <iterator>

namespace ise {

struct Realization {
    DateTime time_mark;
    iisr::Channel channel;
    iisr::Frequency frequency;
    std::vector<complex> quadratures;
};


class DataSource {
public:
    DataSource(list of files);
    DataSource(multicast socket);
    DataSource(directory);

    class DataSourceIterator {
        using value_type = Realization;
        using iterator = std::forward_iterator_tag;
    };

DataSourceIterator begin();
DataSourceIterator end();  // or sentinel

private:
    bytestream stream;
};

}  // namespace
