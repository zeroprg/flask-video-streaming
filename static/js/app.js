
/** @jsx React.DOM */
let API = ""
let DEFAULT_QUERY ="urls?list=true"
let URL = API + "urls"
let deleteURL = URL + "?delete="
var URLlist = React.createClass({

    getInitialState: function () {
        return {
            sortedArr: [],
            regArray: [],
            data: [],
        }
    },

    buttonClickHide(arg1, arg2) {
        document.getElementById("section"+arg2).style.display ='none';
    },

    buttonClickShow(data, cam) {
        console.log("cam:" + cam); 
        console.log("data:" + data);
        elem = document.getElementById("section"+cam);
        elem.style.display ='block'
        window.moreparams(cam, "0-1");
        window.showmore(1,cam,0);
        elem.scrollIntoView();

/*
        fetch(data)
             .then(response => {
                console.log(" response:" + response)
                if (response.ok) {
                    console.log(" response:" + JSON.stringify(response, null, 2) ); 
                    this.loadData();
                    return response.json();
                } else {
                    console.log(" error:")
                    throw new Error('Something went wrong ...');
                }
            })
*/
  },

    loadData() {
        this.setState({ isLoading: true });
        console.log(" start:")
        fetch(API + DEFAULT_QUERY)
            .then(response => {
                console.log(" response:" + response)
                if (response.ok) {
                    //console.log(" response:" + JSON.stringify(response, null, 2) )
                    return response.json();
                } else {
                    console.log(" error:")
                    throw new Error('Something went wrong ...');
                }
            })
            .then(data => this.setState({ data: data, url: deleteURL, isLoading: false }))
            .catch(error => this.setState({ error, isLoading: false }));
    },


    componentDidMount() {
        this.loadData()
    },

    render() {
        const { data, isLoading, error } = this.state;

        if (error) {
            return <p>{error.message}</p>;
        }

        if (isLoading) {
            return <p>Loading ...</p>;
        }

        return (
            <ul>
                {data.map(data =>
                    <li key={data[0]}>
                        <a href={data[1]}>{data[1]}</a>
                        &nbsp;
                        <a onClick={() => this.buttonClickShow(this.state.url+data[1] , data[0])} className="btn btn-primary a-btn - slide - text">
                            <span className="glyphicon glyphicon-play-circle" aria-hidden="true"></span>
                            <span>
                                <strong>Show</strong>
                            </span>
                        </a>
                        <a  onClick={() =>  this.buttonClickHide(this.state.url+data[1] , data[0])} className="btn btn-primary a-btn - slide - text">
                            <span className="glyphicon glyphicon-" aria-hidden="true"></span>
                            <span>
                                <strong>Hide</strong>
                            </span>
                        </a>
                    </li>
                )}
            </ul>
        );
    }

})

React.renderComponent(<URLlist />, document.getElementById('URLlist'));
